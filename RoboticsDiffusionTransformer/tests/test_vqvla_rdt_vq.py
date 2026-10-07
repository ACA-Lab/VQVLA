import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from scripts.vqvla_rdt_vq import (
    RdtMixedWeightRouter,
    dequantize_matrix,
    pack_indices,
    quantize_matrix,
    unpack_indices,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2)
        self.norm = nn.LayerNorm(3)
        self.register_parameter("embedding_table", nn.Parameter(torch.randn(4, 2)))
        self.register_parameter("odd", nn.Parameter(torch.tensor([0.1, -0.2, 0.3])))
        self.register_parameter("scalar", nn.Parameter(torch.tensor(0.7)))


class RdtVqTests(unittest.TestCase):
    def test_packed_indices_round_trip(self):
        for bits, codebook_size in ((6, 64), (8, 256)):
            values = torch.arange(257, dtype=torch.long) % codebook_size
            packed = pack_indices(values, bits)
            self.assertTrue(torch.equal(values, unpack_indices(packed, values.numel(), bits)))

    def test_ragged_matrix_round_trip(self):
        weight = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        archive, reconstructed = quantize_matrix(
            weight,
            torch.ones(6),
            device="cpu",
            block_rows=2,
            block_cols=4,
            codebook_size=4,
            vector_len=2,
            kmeans_iters=2,
            block_batch_size=1,
            seed=5,
        )
        decoded = dequantize_matrix(archive, dtype=weight.dtype, device="cpu")
        self.assertEqual(tuple(decoded.shape), tuple(weight.shape))
        self.assertTrue(torch.allclose(reconstructed, decoded, atol=1e-3, rtol=1e-3))

    def test_router_covers_bias_norm_odd_vector_and_scalar(self):
        model = TinyPolicy()
        original_bias = model.proj.bias.detach().clone()
        original_norm_weight = model.norm.weight.detach().clone()
        modules = {"rdt.proj": model.proj}
        module_weight_ids = {id(module.weight) for module in modules.values()}
        parameters = {
            f"rdt.{name}": parameter
            for name, parameter in model.named_parameters()
            if id(parameter) not in module_weight_ids and parameter.ndim >= 2
        }
        targets = {
            **{
                name: (module.weight, tuple(module.weight.shape), True)
                for name, module in modules.items()
            },
            **{
                name: (parameter, tuple(parameter.shape), False)
                for name, parameter in parameters.items()
            },
        }

        archives = []
        for codebook_size in (256, 64):
            layers = {}
            for index, (name, (parameter, shape, module_weight)) in enumerate(targets.items()):
                if module_weight:
                    matrix = parameter.detach().reshape(shape[0], -1).cpu()
                elif parameter.ndim >= 2:
                    matrix = parameter.detach().reshape(-1, shape[-1]).cpu()
                else:
                    matrix = parameter.detach().reshape(1, -1).cpu()
                    if matrix.shape[1] % 2:
                        matrix = torch.nn.functional.pad(matrix, (0, 1))
                archive, _ = quantize_matrix(
                    matrix,
                    torch.ones(matrix.shape[1]),
                    device="cpu",
                    block_rows=256,
                    block_cols=256,
                    codebook_size=codebook_size,
                    vector_len=2,
                    kmeans_iters=2,
                    block_batch_size=1,
                    seed=index,
                )
                archive["original_parameter_shape"] = shape
                layers[name] = archive
            archives.append({"version": 1, "layers": layers})

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = []
            for index, archive in enumerate(archives):
                path = Path(temporary_directory) / f"{index}.pt"
                torch.save(archive, path)
                paths.append(str(path))
            router = RdtMixedWeightRouter.load({"rdt": model}, paths[0], paths[1])

        self.assertEqual(len(router.routed_modules), 1)
        self.assertEqual(len(router.routed_parameters), len(parameters))
        self.assertEqual(len(router.routed_parameters), 1)
        self.assertTrue(torch.equal(model.norm.weight, original_norm_weight))
        self.assertTrue(torch.equal(model.proj.bias, original_bias))
        self.assertTrue(torch.equal(model.odd, torch.tensor([0.1, -0.2, 0.3])))
        self.assertAlmostEqual(model.scalar.item(), 0.7)
        router.set_execution_state(False)
        self.assertEqual(router.active_bits, 3)
        router.set_execution_state(True)
        self.assertEqual(router.active_bits, 4)


if __name__ == "__main__":
    unittest.main()
