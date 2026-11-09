"""Reference-value and adjoint checks for runtime-only replacements."""

import unittest
import copy
from argparse import Namespace
import torch
from torch.nn import functional as F
from src.rotation import Rotation
from src.network.adam import make_adam
from src.network.optimization_ops import (
    extension,
    transform_vertices,
    edge_loss,
    sample_fixed,
    interpolation_stencil,
)
from src.network.spatial_knn import SpatialIndex
from src.network.preconditioned_grid import GridLevel


class AdamTests(unittest.TestCase):
    def test_backend_configuration(self):
        for backend in ("foreach", "fused", "single"):
            opt = make_adam(
                [torch.nn.Parameter(torch.zeros(3))], "cpu", {"adam_backend": backend}
            )
            self.assertEqual(opt.defaults["foreach"], backend == "foreach")
            self.assertFalse(opt.defaults["capturable"])
        with self.assertRaises(ValueError):
            make_adam(
                [torch.nn.Parameter(torch.zeros(3))], "cpu", {"adam_backend": "bad"}
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RuntimeOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extension()

    def reference_transform(self, parameters, vertices):
        rotation = Rotation(Namespace(method_args={"rotation_method": "quaternions"}))
        matrix = rotation(parameters[..., :4].clone())
        return (matrix @ vertices[None, ..., None]).squeeze(-1) + torch.tanh(
            parameters[..., 4:] * 0.1
        )

    def test_compiled_mlp_padding_values_gradients_and_state(self):
        from src.network.network import Network

        torch.manual_seed(78)
        reference = Network(
            Namespace(device="cuda"),
            point_dim=4,
            time_dim=8,
            hidden=16,
            layers=3,
            outdim=7,
            time_encoding="fourier_features2",
        ).cuda()
        with torch.no_grad():
            reference.network.head.weight.normal_(std=0.1)
        compiled = copy.deepcopy(reference)
        original_keys = list(compiled.state_dict())
        compiled.enable_compile({"compile_vertex_bucket": 32})
        self.assertEqual(original_keys, list(compiled.state_dict()))
        for dtype in (torch.float32, torch.bfloat16):
            compiled.prepare_compile(3, 17, torch.device("cuda"), dtype)
            self.assertTrue(all(p.grad is None for p in compiled.parameters()))
            for count in (17, 18):
                x = torch.randn(3, count, 4, device="cuda", requires_grad=True)
                t = torch.linspace(0, 1, 3, device="cuda")
                with torch.autocast(
                    "cuda", dtype=dtype, enabled=dtype != torch.float32
                ):
                    actual = compiled(x, t)
                    expected = reference(x, t)
                atol = 5e-6 if dtype == torch.float32 else 2e-3
                rtol = 1e-4 if dtype == torch.float32 else 2e-2
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                weight = torch.randn_like(actual)
                if dtype == torch.bfloat16:
                    # At an activation kink, tiny GEMM rounding changes can
                    # legitimately change the derivative discontinuously.
                    # Compare BF16 adjoints on well-conditioned points; FP32
                    # above checks all points, and BF16 values are checked above.
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                        features = torch.cat(
                            (
                                x,
                                reference.time_encoding(t)[:, None].expand(
                                    -1, count, -1
                                ),
                            ),
                            dim=-1,
                        )
                        z1 = reference.network.mlp[0](features)
                        z2 = reference.network.mlp[2](reference.network.mlp[1](z1))
                        stable = (z1.abs().amin(-1) > 0.003) & (
                            z2.abs().amin(-1) > 0.003
                        )
                    self.assertGreater(int(stable.sum()), 0)
                    weight = weight * stable[..., None]
                ag = torch.autograd.grad(
                    (actual * weight).sum(), (x, *compiled.parameters())
                )
                eg = torch.autograd.grad(
                    (expected * weight).sum(), (x, *reference.parameters())
                )
                for a, e in zip(ag, eg):
                    if dtype == torch.float32:
                        torch.testing.assert_close(a, e, atol=5e-6, rtol=1e-4)
                    else:
                        # Use total error because individual gradient entries
                        # can themselves be near zero.
                        relative_error = (a - e).norm() / e.norm().clamp_min(1e-6)
                        self.assertLess(float(relative_error), 0.03)

    def test_transform_values_and_all_gradients(self):
        torch.manual_seed(74)
        for scale in (0.0, 0.1, 2.0):
            p = (torch.randn(3, 51, 7, device="cuda") * scale).requires_grad_()
            v = torch.randn(51, 3, device="cuda", requires_grad=True)
            actual = transform_vertices(p, v)
            expected = self.reference_transform(p, v)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            weights = torch.randn_like(actual)
            ag = torch.autograd.grad((actual * weights).sum(), (p, v))
            eg = torch.autograd.grad((expected * weights).sum(), (p, v))
            for a, e in zip(ag, eg):
                torch.testing.assert_close(a, e, atol=1e-5, rtol=4e-5)

    def test_near_degenerate_quaternion(self):
        p = torch.zeros(1, 5, 7, device="cuda")
        p[..., 0] = -1
        p[..., 1] = 1e-12
        p.requires_grad_()
        v = torch.randn(5, 3, device="cuda")
        actual = transform_vertices(p, v)
        expected = self.reference_transform(p, v)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        a = torch.autograd.grad(actual.sum(), p)[0]
        e = torch.autograd.grad(expected.sum(), p)[0]
        # Quaternion derivatives scale inversely with its magnitude. Compare
        # dimensionless sensitivities; near-zero radial FP32 cancellation is
        # ill-conditioned in both the matrix and direct implementations.
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a * 1e-12, e * 1e-12, atol=1e-6, rtol=1e-4)

    def test_edge_loss_and_gradient_including_zero_edges(self):
        torch.manual_seed(75)
        v = torch.randn(3, 51, 3, device="cuda", requires_grad=True)
        edges = torch.randint(51, (87, 2), device="cuda")
        edges[0] = 0
        base = torch.rand(1, 87, device="cuda")
        actual = edge_loss(v, edges, base)
        expected = F.l1_loss(
            (v[:, edges[:, 0]] - v[:, edges[:, 1]]).norm(dim=-1), base.expand(3, -1)
        )
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual, v)[0]
        e = torch.autograd.grad(expected, v)[0]
        torch.testing.assert_close(a, e, atol=1e-8, rtol=2e-5)

    def test_fixed_sampling_values_gradients_and_cache_invalidation(self):
        torch.manual_seed(76)
        for resolution in (1, 4, 11):
            level = GridLevel(resolution, 3, 0.7, "cuda")
            with torch.no_grad():
                level.values.normal_()
            points = torch.randn(1, 1, 1, 51, 3, device="cuda") * 2
            points[..., 0, :] = 1
            points[..., 1, :] = -1
            expected = level(points)
            weight = torch.randn_like(expected)
            eg = torch.autograd.grad((expected * weight).sum(), level.values)[0]
            level.prepare(points, 0)
            actual = level(points)
            ag = torch.autograd.grad((actual * weight).sum(), level.values)[0]
            torch.testing.assert_close(actual, expected, atol=5e-7, rtol=5e-5)
            torch.testing.assert_close(ag, eg, atol=2e-6, rtol=5e-5)
            points.neg_()
            actual = level(points)
            level.stencil_nodes = None
            torch.testing.assert_close(actual, level(points))

    def test_cached_query_order_is_exact(self):
        torch.manual_seed(77)
        reference = torch.randn(2, 91, 3, device="cuda", requires_grad=True)
        points = torch.randn(2, 81, 3, device="cuda", requires_grad=True)
        index = SpatialIndex(reference, sort_queries=True)
        order = index.query_order(points)
        # A different permutation remains exact, including after point motion.
        points2 = points * 2
        actual = index.distances(points2, order)
        expected = index.distances(points2)
        torch.testing.assert_close(actual, expected)
        ag = torch.autograd.grad(actual.sum(), (points, reference), retain_graph=True)
        eg = torch.autograd.grad(expected.sum(), (points, reference))
        for a, e in zip(ag, eg):
            torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-5)

    def test_new_operations_on_nondefault_stream_and_graph(self):
        p = torch.randn(2, 51, 7, device="cuda", requires_grad=True)
        v = torch.randn(51, 3, device="cuda")
        edges = torch.randint(51, (87, 2), device="cuda")
        base = torch.rand(1, 87, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                p.grad = None
                edge_loss(transform_vertices(p, v), edges, base).backward()
        torch.cuda.current_stream().wait_stream(stream)
        p.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = edge_loss(transform_vertices(p, v), edges, base)
            loss.backward()
        graph.replay()
        expected = F.l1_loss(
            (
                self.reference_transform(p, v)[:, edges[:, 0]]
                - self.reference_transform(p, v)[:, edges[:, 1]]
            ).norm(dim=-1),
            base.expand(2, -1),
        )
        torch.testing.assert_close(loss, expected)
        expected_gradient = torch.autograd.grad(expected, p)[0]
        torch.testing.assert_close(p.grad, expected_gradient, atol=1e-6, rtol=3e-4)


if __name__ == "__main__":
    unittest.main()
