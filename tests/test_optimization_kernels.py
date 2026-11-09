"""Numerical and gradient checks for the two replacement numerical kernels."""

import unittest
from argparse import Namespace

import torch
from pytorch3d.ops import knn_points

from src.network.preconditioned_grid import GridLevel, axis_eigendecomposition
from src.network.spatial_knn import SpatialIndex, extension
from src.loss_time_smoothing import LossTimeSmoothing


class SeparableSolveTests(unittest.TestCase):
    def test_tensor_epoch_matches_original_smoothing_schedule(self):
        for method in ("pdg", "none", "linear", "exponential", "lerp"):
            args = Namespace(
                device="cpu",
                method_args={
                    "keyframe_index": 3,
                    "optimization": {
                        "epochs": 1000,
                        "time_smoothing": "pdg",
                        "catchup_scaling": method,
                    },
                },
            )
            loss = LossTimeSmoothing(args)
            distances = torch.linspace(0.1, 3, 17)
            for epoch in (0, 3, 10, 250, 999):
                torch.testing.assert_close(
                    loss(distances, epoch), loss(distances, torch.tensor(float(epoch)))
                )

    def test_precomputed_sampling_and_adjoint(self):
        torch.manual_seed(26)
        for resolution in (2, 4, 7):
            level = GridLevel(resolution, 3, 0.7, "cpu")
            level.values.data.normal_()
            points = torch.randn(1, 1, 1, 41, 3)
            expected = level(points)
            weight = torch.randn_like(expected)
            expected_gradient = torch.autograd.grad(
                (expected * weight).sum(), level.values
            )[0]
            level.prepare(points, max_nodes=512)
            actual = level(points)
            actual_gradient = torch.autograd.grad(
                (actual * weight).sum(), level.values
            )[0]
            torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
            torch.testing.assert_close(
                actual_gradient, expected_gradient, atol=3e-7, rtol=3e-6
            )
            # New coordinates must use the general path, not the stale cache.
            other = points.clone().neg_()
            level.sampling, cached = None, level.sampling
            expected_other = level(other)
            level.sampling = cached
            torch.testing.assert_close(level(other), expected_other)

    def test_sampling_cache_invalidates_after_inplace_edit(self):
        level = GridLevel(4, 3, 0.7, "cpu")
        with torch.no_grad():
            level.values.normal_()
        points = torch.randn(1, 1, 1, 19, 3)
        level.prepare(points, max_nodes=64)
        points.neg_()
        actual = level(points)
        level.sampling = None
        torch.testing.assert_close(actual, level(points))

    def test_matches_dense_system_and_adjoint(self):
        torch.manual_seed(27)
        for resolution in (1, 2, 4, 7):
            with self.subTest(resolution=resolution):
                level = GridLevel(resolution, 3, 0.7, "cpu")
                level.values.data.normal_()
                eigenvalues, basis = axis_eigendecomposition(resolution)
                axis = basis @ eigenvalues.diag() @ basis.T
                eye = torch.eye(resolution, dtype=torch.float64)
                matrix = torch.eye(resolution**3, dtype=torch.float64) + 0.7 * (
                    torch.kron(torch.kron(axis, eye), eye)
                    + torch.kron(torch.kron(eye, axis), eye)
                    + torch.kron(torch.kron(eye, eye), axis)
                )
                rhs = level.values.detach().double().reshape(-1, 3).requires_grad_()
                expected = torch.linalg.solve(matrix, rhs)
                actual = level.solve().reshape(-1, 3)
                torch.testing.assert_close(
                    actual.double(), expected, atol=3e-7, rtol=3e-6
                )
                weight = torch.randn_like(actual)
                (actual * weight).sum().backward()
                (expected * weight.double()).sum().backward()
                torch.testing.assert_close(
                    level.values.grad.reshape(-1, 3).double(),
                    rhs.grad,
                    atol=3e-7,
                    rtol=3e-6,
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SpatialKNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extension()

    def check_case(self, query, reference):
        query = query.contiguous().cuda().requires_grad_()
        reference = reference.contiguous().cuda().requires_grad_()
        actual = SpatialIndex(reference).distances(query)
        expected = knn_points(query, reference, K=1).dists
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
        weight = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * weight).sum(), (query, reference))
        expected_grads = torch.autograd.grad(
            (expected * weight).sum(), (query, reference)
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-4, rtol=2e-5)

    def test_rejects_invalid_inputs_before_kernel_launch(self):
        for reference in (
            torch.zeros(1, 0, 3, device="cuda"),
            torch.zeros(0, 4, 3, device="cuda"),
            torch.zeros(1, 4, 2, device="cuda"),
            torch.zeros(1, 4, 3, device="cuda", dtype=torch.float16),
        ):
            with self.assertRaises(ValueError):
                SpatialIndex(reference)
        index = SpatialIndex(torch.zeros(1, 4, 3, device="cuda"))
        for points in (
            torch.zeros(2, 3, 3, device="cuda"),
            torch.zeros(1, 3, 2, device="cuda"),
            torch.zeros(1, 3, 3, device="cuda", dtype=torch.float16),
        ):
            with self.assertRaises(RuntimeError):
                index.distances(points)

    def test_random_clouds_and_outside_queries(self):
        torch.manual_seed(29)
        self.check_case(torch.randn(3, 231, 3) * 4, torch.randn(3, 487, 3))

    def test_thin_clouds(self):
        torch.manual_seed(30)
        reference = torch.randn(2, 413, 3)
        reference[..., 1:] *= 1e-5
        self.check_case(torch.randn(2, 191, 3), reference)

    def test_singleton_and_duplicate_clouds(self):
        self.check_case(torch.randn(2, 45, 3), torch.zeros(2, 1, 3))
        self.check_case(torch.randn(2, 45, 3), torch.zeros(2, 100, 3))

    def test_ties_and_cell_boundaries(self):
        reference = torch.tensor(
            [
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                ]
            ]
        )
        self.check_case(
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0625, 0.0, 0.0],
                        [0.0, 0.0, 0.5],
                        [4.0, 3.0, -2.0],
                    ]
                ]
            ),
            reference,
        )

    def test_empty_queries(self):
        result = SpatialIndex(torch.randn(1, 20, 3, device="cuda")).distances(
            torch.empty(1, 0, 3, device="cuda")
        )
        self.assertEqual(result.shape, (1, 0, 1))

    def test_sorted_queries_preserve_values_and_gradients(self):
        torch.manual_seed(32)
        reference = torch.randn(2, 529, 3, device="cuda", requires_grad=True)
        query = torch.randn(2, 417, 3, device="cuda", requires_grad=True)
        actual = SpatialIndex(reference, sort_queries=True).distances(query)
        expected = knn_points(query, reference, K=1).dists
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
        actual_grads = torch.autograd.grad(actual.sum(), (query, reference))
        expected_grads = torch.autograd.grad(expected.sum(), (query, reference))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-4, rtol=2e-5)

    def test_current_stream_and_cuda_graph(self):
        torch.manual_seed(31)
        reference = torch.randn(2, 321, 3, device="cuda")
        query = torch.randn(2, 127, 3, device="cuda", requires_grad=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                query.grad = None
                SpatialIndex(reference, sort_queries=True).distances(
                    query
                ).sum().backward()
        torch.cuda.current_stream().wait_stream(stream)
        query.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = SpatialIndex(reference, sort_queries=True).distances(query)
            actual.sum().backward()
        graph.replay()
        actual_grad = query.grad.clone()
        expected = knn_points(query, reference, K=1).dists
        expected_grad = torch.autograd.grad(expected.sum(), query)[0]
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-4, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
