import dataclasses
import enum
import logging
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Complete GPTVQ archive for the execution-state 4-bit model.
    gptvq_archive: str | None = None
    # Complete GPTVQ archive for the transition-state 3-bit model.
    gptvq_archive_3bit: str | None = None
    # Select the transition-state model when the prior action metric is at least this value.
    routing_threshold: float | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _create_base_policy(args: Args) -> _policy.Policy:
    """Create an unmodified policy from the selected checkpoint."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _load_complete_archive(policy: _policy.Policy, archive: str) -> None:
    from openpi.quantization.gptvq_archive import load_gptvq_archive

    coverage = load_gptvq_archive(policy._model, archive)  # noqa: SLF001
    logging.info("Loaded complete GPTVQ archive %s (%d parameter values)", archive, coverage["floating_value_count"])


class TransitionStatePolicy:
    """Route whole-policy calls between execution- and transition-state models."""

    def __init__(self, model1: _policy.Policy, model2: _policy.Policy, threshold: float) -> None:
        self._model1 = model1
        self._model2 = model2
        self._threshold = threshold
        self._previous_translation = float("-inf")

    @property
    def metadata(self) -> dict:
        return self._model1.metadata

    def infer(self, observation: dict) -> dict:
        use_transition_state = self._previous_translation >= self._threshold
        result = (self._model2 if use_transition_state else self._model1).infer(observation)
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[1] < 3:
            raise ValueError(f"Expected an action chunk with XYZ columns, got {actions.shape}")
        self._previous_translation = float(np.linalg.norm(actions[:, :3], axis=-1).mean())
        result["vqvla_routing_metric"] = self._previous_translation
        result["vqvla_route"] = "transition_state_3bit" if use_transition_state else "execution_state_4bit"
        return result


def create_policy(args: Args) -> _policy.Policy | TransitionStatePolicy:
    """Create an original, a complete-VQ, or a two-model routed policy."""
    if (args.gptvq_archive_3bit is None) != (args.routing_threshold is None):
        raise ValueError("--gptvq-archive-3bit and --routing-threshold must be provided together")
    policy = _create_base_policy(args)
    if args.gptvq_archive is not None:
        _load_complete_archive(policy, args.gptvq_archive)
    if args.gptvq_archive_3bit is not None:
        if args.gptvq_archive is None:
            raise ValueError("--gptvq-archive is required with --gptvq-archive-3bit")
        model2 = _create_base_policy(args)
        _load_complete_archive(model2, args.gptvq_archive_3bit)
        return TransitionStatePolicy(policy, model2, args.routing_threshold)
    return policy


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
