"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from vqvla.gptvq_archive import load_complete_archive

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _inference_dtype() -> torch.dtype:
    """Select a portable inference dtype without silently changing integers."""
    dtype_name = os.environ.get("OPENVLA_DTYPE", "float16")
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype_name not in dtypes:
        raise ValueError(f"OPENVLA_DTYPE must be one of {sorted(dtypes)}, got {dtype_name!r}")
    return dtypes[dtype_name]


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    attention_implementation = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager")
    inference_dtype = _inference_dtype()
    print(f"[*] Loading in {inference_dtype} with attention implementation: {attention_implementation}")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation=attention_implementation,
        torch_dtype=inference_dtype,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


class TransitionStateRouter:
    """Route action queries between complete 4bit and 3bit OpenVLA policies."""

    def __init__(self, execution_model, transition_model, threshold: float):
        self.execution_model = execution_model
        self.transition_model = transition_model
        self.threshold = threshold
        self.norm_stats = execution_model.norm_stats
        self._previous_action: np.ndarray | None = None
        self.execution_queries = 0
        self.transition_queries = 0
        self.metrics: list[float] = []

    def reset_action_history(self) -> None:
        self._previous_action = None

    def predict_action(self, **kwargs):
        metric = float("inf") if self._previous_action is None else float(np.linalg.norm(self._previous_action[:3]))
        # An episode starts without a robot-action history.  Keep that
        # bootstrap query on the execution-state model; otherwise ``inf``
        # would accidentally select the transition-state model.
        use_transition = self._previous_action is not None and metric >= self.threshold
        selected = self.transition_model if use_transition else self.execution_model
        action = selected.predict_action(**kwargs)
        self._previous_action = np.asarray(action, dtype=np.float32).copy()
        self.metrics.append(metric)
        if use_transition:
            self.transition_queries += 1
        else:
            self.execution_queries += 1
        return action

    def routing_summary(self) -> dict[str, float | int]:
        total = self.execution_queries + self.transition_queries
        finite_metrics = [metric for metric in self.metrics if np.isfinite(metric)]
        return {
            "execution_state_4bit_queries": self.execution_queries,
            "transition_state_3bit_queries": self.transition_queries,
            "transition_state_ratio": self.transition_queries / total if total else 0.0,
            "metric_p50": float(np.percentile(finite_metrics, 50)) if finite_metrics else float("nan"),
            "metric_p90": float(np.percentile(finite_metrics, 90)) if finite_metrics else float("nan"),
        }


def load_vqvla_policy(cfg):
    """Load source OpenVLA and replace every parameter from GPTVQ archive(s)."""
    execution_model = get_vla(cfg)
    if not cfg.gptvq_archive_4bit:
        return execution_model
    execution_coverage = load_complete_archive(execution_model, cfg.gptvq_archive_4bit)
    print(f"[*] Loaded complete execution-state 4bit archive: {execution_coverage}")
    if not cfg.gptvq_archive_3bit:
        return execution_model
    transition_model = get_vla(cfg)
    transition_coverage = load_complete_archive(transition_model, cfg.gptvq_archive_3bit)
    print(f"[*] Loaded complete transition-state 3bit archive: {transition_coverage}")
    return TransitionStateRouter(execution_model, transition_model, cfg.routing_threshold)


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    parameter_model = vla.execution_model if isinstance(vla, TransitionStateRouter) else vla
    inputs = processor(prompt, image).to(DEVICE, dtype=next(parameter_model.parameters()).dtype)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
