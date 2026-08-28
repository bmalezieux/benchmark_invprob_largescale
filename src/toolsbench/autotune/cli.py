"""The two subcommands.

    toolsbench autotune seam --config C
    toolsbench autotune plan --config C --overlap 32 --gpus 4 --gpu-mem 32

`seam` prints what it measured and picks nothing; `plan` probes the config and
reports what fits at the overlap you picked. See the package docstring for the
memory model and for why the stages are separate.
"""

from __future__ import annotations

import argparse

from toolsbench.autotune.config import BenchCase, parse_config
from toolsbench.autotune.probe import identity_run
from toolsbench.autotune.seam import SEAM_SIGMA, load_cache, save_cache, seam_floors
from toolsbench.autotune.select import rank, top3
from toolsbench.utils import create_denoiser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolsbench autotune",
        description="Suggest tiling parameters for a distributed config.",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    for name, help_text in (
        ("seam", "stage A: measure the denoiser's overlap floor and show it"),
        ("plan", "stage B: rank tilings that fit, for a chosen overlap"),
    ):
        sub = stages.add_parser(name, help=help_text)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument("--config", help="benchopt config to read the problem from")
        source.add_argument(
            "--image", help="spatial size, e.g. '4096' or '512,512,512'"
        )
        sub.add_argument("--ndim", type=int, default=None, choices=(2, 3))
        sub.add_argument("--channels", type=int, default=None)
        sub.add_argument("--denoiser", default=None)
        sub.add_argument(
            "--sigma",
            type=float,
            default=None,
            help="denoiser sigma; the floor tracks it, so it is part "
            "of the cache key (default: denoiser_sigma from the config, else 0.01)",
        )
        sub.add_argument("--device", default="cuda")

    seam = stages.choices["seam"]
    seam.add_argument(
        "--no-cache",
        action="store_true",
        help="re-measure even if a cached result exists",
    )
    seam.add_argument(
        "--thresholds",
        default="0.1,0.5,1.5",
        help="dB below the plateau to report suggestions for; the first is the "
        "baseline, the rest show what looser tolerances would buy",
    )

    plan = stages.choices["plan"]
    plan.add_argument(
        "--overlap", type=int, required=True, help="the overlap you chose from stage A"
    )
    plan.add_argument(
        "--gpus", type=int, required=True, help="world size of the run you are planning"
    )
    plan.add_argument(
        "--gpu-mem", type=float, required=True, help="memory per target GPU, in GB"
    )
    plan.add_argument(
        "--train",
        action="store_true",
        help="unrolled training (inferred from the solver with --config)",
    )
    plan.add_argument("--n-iter", type=int, default=None)
    plan.add_argument(
        "--probe-iter",
        type=int,
        default=2,
        help="iterations the identity-denoiser probe runs (default: 2)",
    )
    plan.add_argument(
        "--probe-peak",
        type=float,
        default=None,
        help="skip the probe: peak MB it would have measured",
    )
    plan.add_argument(
        "--probe-resident",
        type=float,
        default=None,
        help="with --probe-peak: the MB resident between bursts (default: same)",
    )
    plan.add_argument(
        "--calls",
        type=int,
        default=None,
        help="tiled denoiser calls per step; counted by the probe otherwise",
    )
    plan.add_argument("--margin", type=float, default=0.90)
    plan.add_argument(
        "--all",
        action="store_true",
        help="print every feasible candidate, not just three",
    )
    return parser


def _cases(args):

    if args.config:
        return parse_config(args.config, ndim=args.ndim, channels=args.channels)
    dims = [int(v) for v in args.image.split(",")]
    ndim = args.ndim or (3 if len(dims) == 3 else 2)
    if len(dims) == 1:
        dims = dims * ndim
    return [
        BenchCase(
            solver="(command line)",
            denoiser=args.denoiser or "drunet",
            img_size=(1, args.channels or 3, *dims),
            train=getattr(args, "train", False),
            n_iter=getattr(args, "n_iter", None),
            config_patch=None,
            config_overlap=None,
            sigma=args.sigma,
        )
    ]


def _sigma_for(case, args) -> float:
    """Command line, then the config's denoiser_sigma, then the default.

    Worth the indirection: the seam floor tracks the denoising strength, and the
    configs here span 0.005 to 0.05. Measured on the same denoiser and content,
    that range moved the floor by a factor of four.
    """
    if args.sigma is not None:
        return args.sigma
    return float(case.sigma) if case.sigma is not None else SEAM_SIGMA


def run_seam(args) -> int:

    thresholds = tuple(float(v) for v in args.thresholds.split(","))
    cache = {} if args.no_cache else load_cache()
    seen = set()
    for case in _cases(args):
        sigma = _sigma_for(case, args)
        key = f"{case.denoiser}-{case.ndim}d-c{case.img_size[1]}-s{sigma:g}"
        if key in seen:
            continue
        seen.add(key)
        print(f"\n{case.denoiser} {case.ndim}D, {case.img_size[1]}ch, sigma={sigma:g}")
        if key in cache:
            print(cache[key].table(thresholds))
            print("  (cached; --no-cache to re-measure)")
            continue
        model = create_denoiser(
            case.denoiser,
            (1, case.img_size[1], *([64] * case.ndim)),
            device=args.device,
        )
        result = seam_floors(
            model,
            case.denoiser,
            case.ndim,
            case.img_size[1],
            device=args.device,
            sigma=sigma,
        )
        save_cache(result)
        print(result.table(thresholds))
        print(
            "\n  choose an overlap, then: toolsbench autotune plan "
            "--overlap R --gpus N --gpu-mem G ..."
        )
    return 0


def run_plan(args) -> int:

    status = 0
    for case in _cases(args):
        print(
            f"\n{case.describe()} · {args.gpus} GPUs · {args.gpu_mem:g} GB "
            f"· overlap {args.overlap}"
        )
        model = create_denoiser(
            case.denoiser,
            (1, case.img_size[1], *([64] * case.ndim)),
            device=args.device,
        )
        if args.probe_peak is not None:
            peak = args.probe_peak
            resident = args.probe_resident if args.probe_resident is not None else peak
            n_calls, how = (args.calls or 1), "given"
        elif not case.dataset:
            raise SystemExit(
                "the probe rebuilds the benchmark's own objects, so it needs a "
                "config that names a dataset. With --image, pass the numbers "
                "instead: --probe-peak MB [--probe-resident MB] [--calls N]."
            )
        else:
            result = identity_run(
                case,
                world_size=args.gpus,
                n_iter=args.probe_iter,
                device=args.device,
            )
            peak, resident = result.peak_mb, result.resident_mb
            n_calls, how = (args.calls or result.n_calls), "measured"
        print(
            f"  probe: peak {peak:.0f} MB, resident {resident:.0f} MB, "
            f"{n_calls} denoiser call(s) per step ({how})"
        )

        cands = rank(
            case.img_size,
            args.gpus,
            args.gpu_mem,
            overlap=args.overlap,
            train=case.train,
            n_calls=n_calls,
            model_obj=model,
            probe_peak=peak,
            probe_resident=resident,
            margin=args.margin,
            device=args.device,
        )
        if not cands:
            print(
                "  nothing fits: every patch size is over budget, or its single "
                "tile does not fit on this device"
            )
            status = 1
            continue
        for candidate in cands if args.all else top3(cands):
            print(f"    {candidate.as_yaml_row()}")
        if case.config_patch:
            print(
                f"  config currently pins patch_size={case.config_patch}, "
                f"overlap={case.config_overlap}"
            )
    return status


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_seam(args) if args.stage == "seam" else run_plan(args)
