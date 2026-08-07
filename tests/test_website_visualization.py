from __future__ import annotations

import pandas as pd
import pytest

from toolsbench.visualization.comm import Section
from toolsbench.visualization.website.comm import _bandwidth_gbs, _melt_sections
from toolsbench.visualization.website.compile_speedup import (
    _denoiser_speedup_payload,
    _pnp_compile_modes_payload,
    _roofline_payload,
)
from toolsbench.visualization.website.inference_scaling import (
    _overlap_work_multiplier,
    _quality_preservation_payload,
    _summarize_communication,
)
from toolsbench.visualization.website.training_scaling import (
    _batch_size_payload,
    _checkpointing_payload,
    _strong_scaling_payload,
    _weak_scaling_payload,
)

# --- inference_scaling ---


@pytest.mark.parametrize(
    ("image_size", "expected"),
    [(2048, 1.5625), (4096, 1.5625), (8192, 1.41015625)],
)
def test_overlap_work_multiplier_matches_deepinv_grid(image_size, expected):
    multiplier = _overlap_work_multiplier(
        image_size=image_size,
        patch_size=448,
        overlap=32,
        dimensions=2,
    )
    assert multiplier == pytest.approx(expected)


def test_quality_preservation_requires_single_process_reference():
    rows = []
    for image_size, gpu_values in {
        2048: {1: [10.0, 20.0], 2: [10.01, 19.99]},
        8192: {2: [11.0, 21.0], 4: [11.0, 21.02]},
    }.items():
        for gpu_count, psnr_values in gpu_values.items():
            for iteration, psnr in enumerate(psnr_values):
                rows.append(
                    {
                        "p_dataset_image_size": image_size,
                        "n_gpus": gpu_count,
                        "stop_val": iteration,
                        "objective_psnr": psnr,
                        "p_solver_distribute_denoiser": not (
                            image_size == 2048 and gpu_count == 1
                        ),
                    }
                )

    payload = _quality_preservation_payload(pd.DataFrame(rows), {})

    assert payload["methodology"]["baselineGpuCountByImageSize"] == {
        "2048": 1,
    }
    assert [value["psnrDifferenceDb"] for value in payload["values"]] == [
        0.01,
        -0.01,
    ]


def test_communication_summary_excludes_initial_iterations():
    rows = []
    for stop_val, total, gradient, denoise, comm in [
        (1, 100.0, 40.0, 60.0, 20.0),
        (2, 100.0, 40.0, 60.0, 20.0),
        (3, 10.0, 4.0, 6.0, 2.0),
        (4, 14.0, 6.0, 8.0, 4.0),
    ]:
        rows.append(
            {
                "n_gpus": 2,
                "n_nodes": 1,
                "p_solver_distribute_physics": True,
                "p_solver_distribute_denoiser": True,
                "stop_val": stop_val,
                "objective_total_time_sec": total,
                "objective_gradient_cuda_sec": gradient,
                "objective_denoise_cuda_sec": denoise,
                "objective_gradient_comm_sec": comm / 2,
                "objective_denoise_comm_sec": comm / 2,
                "objective_comm_cuda_sec": comm,
                "objective_comm_sync_sec": 0.5,
            }
        )

    assert _summarize_communication(pd.DataFrame(rows), "test") == [
        {
            "problem": "test",
            "gpuCount": 2,
            "nodeCount": 1,
            "mode": "distributed",
            "iterationWallSec": 12.0,
            "computeCudaSec": 9.0,
            "communicationCudaSec": 3.0,
            "synchronizationCudaSec": 0.5,
            "communicationSharePct": 25.0,
            "computeSpeedup": 1.0,
            "idealComputeSpeedup": 1.0,
        }
    ]


# --- compile_speedup ---


def _denoiser_summary_row(**overrides):
    row = {
        "gpu": "h100",
        "denoiser": "dncnn",
        "shape": "512",
        "dim": 2,
        "area": 512 * 512,
        "eager_sec": 0.02,
        "compiled_sec": 0.01,
        "speedup": 2.0,
        "arith_intensity": 68.4,
    }
    row.update(overrides)
    return row


def test_denoiser_speedup_payload_reports_eager_compiled_and_speedup():
    summary = pd.DataFrame(
        [
            _denoiser_summary_row(gpu="a100", speedup=1.5),
            _denoiser_summary_row(gpu="h100", speedup=2.0),
        ]
    )

    payload = _denoiser_speedup_payload(summary, {})

    assert payload["id"] == "compile-denoiser-speedup"
    assert [value["gpu"] for value in payload["values"]] == ["a100", "h100"]
    assert [value["speedup"] for value in payload["values"]] == [1.5, 2.0]
    assert payload["values"][0]["eagerSec"] == 0.02
    assert payload["values"][0]["compiledSec"] == 0.01


def test_roofline_payload_filters_to_requested_gpu_only():
    summary = pd.DataFrame(
        [
            _denoiser_summary_row(gpu="a100", denoiser="dncnn", speedup=1.3),
            _denoiser_summary_row(gpu="h100", denoiser="dncnn", speedup=2.0),
            _denoiser_summary_row(gpu="h100", denoiser="unet", speedup=2.4),
        ]
    )

    payload = _roofline_payload(summary, {}, gpu="h100")

    assert payload["id"] == "compile-denoiser-roofline-h100"
    assert {value["denoiser"] for value in payload["values"]} == {"dncnn", "unet"}
    assert all("arithIntensityFlopByte" in value for value in payload["values"])
    assert payload["methodology"]["gpu"] == "h100"
    assert payload["methodology"]["ridgePointFlopByte"] == pytest.approx(
        494.0 / 3.35, rel=1e-6
    )


def test_pnp_compile_modes_payload_orders_by_gpu_then_compile_mode():
    summary = pd.DataFrame(
        [
            {
                "compile_mode": "fused",
                "n_gpus": 1,
                "first_iter_mean": 5.0,
                "first_iter_std": 0.1,
                "stable_iter_mean": 0.4,
                "stable_iter_std": 0.01,
                "speedup": 1.4,
                "compile_order": 3,
            },
            {
                "compile_mode": "None",
                "n_gpus": 1,
                "first_iter_mean": 1.0,
                "first_iter_std": 0.05,
                "stable_iter_mean": 0.5,
                "stable_iter_std": 0.01,
                "speedup": 1.0,
                "compile_order": 0,
            },
        ]
    )

    payload = _pnp_compile_modes_payload(summary, {})

    assert payload["id"] == "compile-pnp-modes"
    assert [value["compileMode"] for value in payload["values"]] == ["None", "fused"]
    assert payload["values"][0]["stableIterSec"] == 0.5
    assert "pre" in payload["methodology"]["compileModes"]


# --- communication ---


def test_melt_sections_computes_pop_fields():
    section = Section("denoiser", "cuda", "compute", "compute_max", "transfer", "wait")
    summary = pd.DataFrame(
        [
            {
                "n_gpus": 4,
                "n_nodes": 1,
                "payload_bytes": 1_000_000.0,
                "compute": 10.0,
                "compute_max": 11.0,
                "transfer": 0.05,
                "wait": 0.5,
                "cuda": 10.55,
                "comm_share_scale": 0.5,
                "lb_combined_pct": 82.5,
                "comm_e_pct": 91.3,
            }
        ]
    )

    [row] = _melt_sections(summary, [section], study="inference", dim="2D")

    assert row["study"] == "inference"
    assert row["dim"] == "2D"
    assert row["section"] == "denoiser"
    assert row["gpuCount"] == 4
    assert row["computeSec"] == 10.0
    assert row["transferSec"] == 0.05
    assert row["waitSec"] == 0.5
    assert row["lbPct"] == pytest.approx(100 * 10.0 / 11.0)
    assert row["waitPct"] == pytest.approx(5.0)
    # scale is applied, not just passed through
    assert row["scaledComputeSec"] == pytest.approx(5.0)
    assert row["scaledCommSec"] == pytest.approx((0.05 + 0.5) * 0.5)
    # study/dim-combined POP metrics are passed through as-is (computed
    # upstream, across all sections, before melting)
    assert row["lbCombinedPct"] == pytest.approx(82.5)
    assert row["commEPct"] == pytest.approx(91.3)
    # denoiser has exactly one collective (no p_solver_n_iter column), so
    # bandwidth is derivable: 2 * payload * (gpus-1)/gpus bytes / transfer.
    assert row["bandwidthGBs"] == pytest.approx(0.03)


def test_bandwidth_is_none_for_single_gpu_and_undefined_collective_count():
    forward = Section(
        "forward", "cuda_f", "compute_f", "compute_max_f", "transfer_f", "wait_f"
    )
    backward = Section(
        "backward", "cuda_b", "compute_b", "compute_max_b", "transfer_b", "wait_b"
    )
    summary = pd.DataFrame(
        [
            {  # multi-GPU: forward's collective count is derivable, backward's isn't
                "n_gpus": 4,
                "n_nodes": 1,
                "payload_bytes": 1_000_000.0,
                "p_solver_n_iter": 5,
                "comm_share_scale": 1.0,
                "lb_combined_pct": 95.0,
                "comm_e_pct": 90.0,
                "cuda_f": 10.0,
                "compute_f": 9.0,
                "compute_max_f": 9.5,
                "transfer_f": 0.1,
                "wait_f": 0.9,
                "cuda_b": 5.0,
                "compute_b": 4.5,
                "compute_max_b": 4.8,
                "transfer_b": 0.05,
                "wait_b": 0.45,
            },
            {  # single GPU: nothing was transferred, regardless of section
                "n_gpus": 1,
                "n_nodes": 1,
                "payload_bytes": 1_000_000.0,
                "p_solver_n_iter": 5,
                "comm_share_scale": 0.25,
                "lb_combined_pct": 100.0,
                "comm_e_pct": 100.0,
                "cuda_f": 8.0,
                "compute_f": 8.0,
                "compute_max_f": 8.0,
                "transfer_f": 0.0,
                "wait_f": 0.0,
                "cuda_b": 4.0,
                "compute_b": 4.0,
                "compute_max_b": 4.0,
                "transfer_b": 0.0,
                "wait_b": 0.0,
            },
        ]
    )

    rows = _melt_sections(summary, [forward, backward], study="training", dim="2D")
    by_key = {(row["gpuCount"], row["section"]): row for row in rows}

    assert by_key[(4, "forward")]["bandwidthGBs"] is not None
    assert by_key[(4, "backward")]["bandwidthGBs"] is None
    assert by_key[(1, "forward")]["bandwidthGBs"] is None


def test_bandwidth_gbs_matches_ring_allreduce_byte_accounting():
    row = pd.Series({"payload_bytes": 1_000_000.0})

    bandwidth = _bandwidth_gbs(row, "denoiser", gpu_count=4, transfer=0.05)

    # count=1 (no p_solver_n_iter): 2 * 1e6 * 3/4 bytes / 0.05 s / 1e9
    assert bandwidth == pytest.approx(0.03)


# --- training_scaling ---


def _training_summary_rows() -> pd.DataFrame:
    """Two weak-scaling ladders' worth of distributed training summary rows."""
    rows = []
    for image_size, mpix, gpu_counts in [
        (1024, 1.048576, [1, 2]),
        (4096, 16.777216, [1, 4, 16]),
    ]:
        for gpu_count in gpu_counts:
            rows.append(
                {
                    "training_image_size": image_size,
                    "training_mpix": mpix,
                    "n_gpus": gpu_count,
                    "p_solver_distribute_model": True,
                    "avg_total_time_sec": 100.0 / gpu_count,
                    "avg_forward_time_sec": 25.0 / gpu_count,
                    "avg_backward_time_sec": 75.0 / gpu_count,
                    "max_gpu_mb": 1000.0 * gpu_count,
                    "max_forward_gpu_mb": 500.0 * gpu_count,
                }
            )
    return pd.DataFrame(rows)


def test_strong_scaling_efficiency_is_relative_to_each_image_size_baseline():
    payload = _strong_scaling_payload(_training_summary_rows(), {})

    by_key = {
        (value["imageSize"], value["gpuCount"]): value for value in payload["values"]
    }
    # 1024 baselines on 1 GPU, 4096 also on 1 GPU: both start at 100%.
    assert by_key[("1024x1024", 1)]["efficiencyPct"] == 100.0
    assert by_key[("1024x1024", 2)]["efficiencyPct"] == 100.0
    assert by_key[("4096x4096", 16)]["speedup"] == 16.0
    assert payload["methodology"]["baselineByImageSize"]["4096x4096"]["gpuCount"] == 1


def test_weak_scaling_ladders_group_runs_with_the_same_per_gpu_workload():
    payload = _weak_scaling_payload(_training_summary_rows(), {})

    ladders = {value["localWorkloadMpix"] for value in payload["values"]}
    # 1024 on 1 GPU and 4096 on 16 GPUs are both ~1.05 Mpix per rank.
    assert 1.05 in ladders
    ladder = [
        value for value in payload["values"] if value["localWorkloadMpix"] == 1.05
    ]
    assert {value["gpuCount"] for value in ladder} == {1, 16}
    # Ladders with a single point carry no information and are dropped.
    assert 16.78 not in ladders


def test_checkpointing_marks_configurations_that_do_not_fit():
    summary = pd.DataFrame(
        [
            {
                "p_solver_checkpoint_batches": "always",
                "n_gpus": 16,
                "p_solver_max_batch_size": 1,
                "avg_total_time_sec": 1.9,
                "avg_forward_time_sec": 0.5,
                "avg_backward_time_sec": 1.4,
                "max_gpu_mb": 4674.0,
                "max_forward_gpu_mb": 2001.0,
                "max_backward_gpu_mb": 4674.0,
            }
        ]
    )

    payload = _checkpointing_payload(summary, {})

    measured = [value for value in payload["values"] if value["status"] == "measured"]
    infeasible = [
        value for value in payload["values"] if value["status"] == "did-not-fit"
    ]
    assert len(measured) == 1
    assert {value["gpuCount"] for value in infeasible} == {1, 4}
    # Declared-infeasible entries carry no fabricated measurements.
    assert all("peakMemoryMb" not in value for value in infeasible)


def test_checkpointing_rejects_a_declared_infeasible_run_that_was_measured():
    summary = pd.DataFrame(
        [
            {
                "p_solver_checkpoint_batches": "never",
                "n_gpus": 1,
                "p_solver_max_batch_size": 1,
                "avg_total_time_sec": 24.7,
                "avg_forward_time_sec": 5.7,
                "avg_backward_time_sec": 19.0,
                "max_gpu_mb": 6308.0,
                "max_forward_gpu_mb": 3499.0,
                "max_backward_gpu_mb": 6308.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="declared infeasible"):
        _checkpointing_payload(summary, {})


def test_batch_size_efficiency_uses_one_shared_baseline_not_one_per_gpu_count():
    """A per-GPU-count baseline would pin every curve to 100% and hide the effect."""
    summary = pd.DataFrame(
        [
            {
                "n_gpus": gpu_count,
                "p_solver_max_batch_size": batch,
                # batch 4 is twice as fast as batch 1 at every GPU count
                "avg_total_time_sec": (100.0 if batch == 1 else 50.0) / gpu_count,
                "max_gpu_mb": 1024.0 if batch == 1 else 2048.0,
                "max_forward_gpu_mb": 512.0,
            }
            for gpu_count in (1, 4)
            for batch in (1, 4)
        ]
    )

    payload = _batch_size_payload(summary, {})

    by_key = {
        (value["maxBatchSize"], value["gpuCount"]): value for value in payload["values"]
    }
    # Baseline is the largest batch at the fewest GPUs, shared by every series.
    assert payload["methodology"]["baseline"]["maxBatchSize"] == 4
    assert payload["methodology"]["baseline"]["gpuCount"] == 1
    assert by_key[(4, 1)]["efficiencyPct"] == 100.0
    # Batch 1 stays visibly below it instead of being renormalized to 100%.
    assert by_key[(1, 1)]["efficiencyPct"] == 50.0
    assert by_key[(1, 4)]["efficiencyPct"] == 50.0
    assert by_key[(4, 4)]["efficiencyPct"] == 100.0
    assert by_key[(4, 1)]["peakMemoryGb"] == 2.0
