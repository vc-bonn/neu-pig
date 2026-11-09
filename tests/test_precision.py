import unittest

import torch

from src.network.precision import compute_dtype
from src.network.preconditioned_grid import GridLevel


class PrecisionConfigurationTests(unittest.TestCase):
    def test_validates_precision_names(self):
        self.assertEqual(compute_dtype("fp16"), torch.float16)
        self.assertEqual(compute_dtype("bf16"), torch.bfloat16)
        self.assertEqual(compute_dtype("fp32"), torch.float32)
        with self.assertRaises(ValueError):
            compute_dtype("int8")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MixedPrecisionGridTests(unittest.TestCase):
    def test_values_and_scaled_gradients_against_fp32(self):
        torch.manual_seed(43)
        for dtype in (torch.float16, torch.bfloat16):
            for resolution, max_nodes in ((4, 64), (11, 0)):
                with self.subTest(dtype=dtype, resolution=resolution):
                    reference = GridLevel(resolution, 3, 0.7, "cuda")
                    mixed = GridLevel(resolution, 3, 0.7, "cuda", dtype=dtype)
                    with torch.no_grad():
                        reference.values.normal_(std=0.1)
                        mixed.values.copy_(reference.values)
                    points = torch.randn(1, 1, 1, 83, 3, device="cuda")
                    reference.prepare(points, max_nodes)
                    mixed.prepare(points, max_nodes)
                    expected = reference(points)
                    actual = mixed(points)
                    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-2)
                    weight = torch.randn_like(expected)
                    (expected * weight).sum().backward()
                    ((actual * weight).sum() * 128).backward()
                    mixed.values.grad.div_(128)
                    torch.testing.assert_close(
                        mixed.values.grad, reference.values.grad, atol=3e-3, rtol=3e-2
                    )
                    self.assertTrue(torch.isfinite(mixed.values.grad).all())
                    self.assertEqual(mixed.values.dtype, torch.float32)
                    self.assertEqual(mixed.values.grad.dtype, torch.float32)
                    self.assertEqual(mixed.compute_basis.dtype, dtype)
                    if mixed.sampling is not None:
                        self.assertEqual(mixed.sampling.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
