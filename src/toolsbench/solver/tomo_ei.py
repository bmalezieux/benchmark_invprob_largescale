"""Self-supervised cryo-ET training solver (``tomo_ei``).

One benchopt iteration is one training step of demo_cyo's ``tomo_ei`` preset:
each half-set's FBP volume is denoised, the two reconstructions are scored
against the *other* half's measurements, and — when ``eq_weight`` is set — an
equivariance term re-projects a rotated reconstruction through the real
geometry. No ground truth is used by the loss; the synthetic one only exists so
the problem is reproducible.

Standalone algorithm class, no benchopt dependency: instantiate with all
config, call ``run(cb)``, then ``get_result()``.
"""

import torch
from deepinv.distributed import DistributedContext, distribute

from toolsbench.utils.cryo import (
    GpuFSC,
    Rotate3D,
    build_cryo_pair,
    build_unet3d,
    eq_loss,
    fsc_resolution,
    obs_loss,
    split_sinogram,
)
from toolsbench.utils.solver_utils import distributed_callback_iter, sync_and_barrier

# The spelling demo_cyo's configs use. None means "off": no autocast context is
# entered at all, no scaler, no cast anywhere.
AMP_DTYPES = {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}


class TomoEISolver:
    """Half-set equivariant-imaging training, one step per benchopt iteration."""

    def __init__(
        self,
        problem,
        device,
        profiler,
        ctx,
        distributed_mode,
        *,
        tomography_backend="auto",
        num_operators=None,
        eq_weight=0.0,
        learning_rate=1e-4,
        grad_clip=1.0,
        f_maps=64,
        num_levels=4,
        compile_model=False,
        distribute_model=False,
        patch_size=128,
        overlap=16,
        max_batch_size=1,
        checkpoint_batches="auto",
        fsc_threshold=0.143,
        pixel_size=1.0,
        mixed_precision="off",
    ):
        self.problem = problem
        self.device = device
        self.profiler = profiler
        self.ctx = ctx
        self.distributed_mode = distributed_mode
        self.tomography_backend = tomography_backend
        self.num_operators = num_operators
        self.eq_weight = eq_weight
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.f_maps = f_maps
        self.num_levels = num_levels
        self.compile_model = compile_model
        self.distribute_model = distribute_model
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_batch_size = max_batch_size
        self.checkpoint_batches = checkpoint_batches
        self.fsc_threshold = fsc_threshold
        self.pixel_size = pixel_size
        self.mixed_precision = mixed_precision
        if mixed_precision not in AMP_DTYPES:
            raise ValueError(
                f"mixed_precision must be one of {list(AMP_DTYPES)}, "
                f"got {mixed_precision!r}."
            )
        self._amp_dtype = AMP_DTYPES[mixed_precision]
        self._amp_device = torch.device(device).type
        # fp16 only: its gradients underflow without a loss multiply. bf16
        # shares fp32's exponent range, so scaling would buy nothing and it was
        # that 65536x multiply that overflowed at native resolution in demo_cyo.
        self._scaler = (
            torch.amp.GradScaler(self._amp_device)
            if mixed_precision == "fp16"
            else None
        )
        self.amp_skipped = 0

        self.module = None
        self.reconstruction = None
        self.fsc_res = None
        self.fsc_shell = None
        self.loss = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_physics(self):
        """Build both half-set operators, sharded when ``num_operators`` asks.

        The measurements are kept as the solver sees them — split to match the
        shards when sharded — since the Obs loss compares ``A(x)`` against them
        shard for shard.
        """
        pair = build_cryo_pair(
            self.problem.physics,
            self.problem.measurements,
            self.device,
            ctx=self.ctx,
            num_operators=self.num_operators,
            backend=self.tomography_backend,
        )
        y_evn, y_odd = (m.to(self.device) for m in self.problem.measurements)
        if pair.num_operators is not None:
            y_evn = split_sinogram(y_evn, pair.num_operators)
            y_odd = split_sinogram(y_odd, pair.num_operators)
        return pair, y_evn, y_odd

    def _setup_model(self):
        """The denoiser to call, the module that owns its parameters, and a label.

        ``distribute`` returns a tiling wrapper that is not an ``nn.Module``:
        it forwards the call but has no ``parameters()`` / ``train()``. The
        undistributed module is therefore kept alongside it, for the optimiser,
        the gradient clipping and the train/eval switch.
        """
        model, info = build_unet3d(
            self.device, f_maps=self.f_maps, num_levels=self.num_levels
        )
        if self.compile_model:
            model = torch.compile(model)
        module = model
        if self.distribute_model and self.ctx is not None:
            # Tiles the denoiser across ranks. tiling_dims are the three
            # spatial axes of the (B, C, D, H, W) volume.
            model = distribute(
                model,
                self.ctx,
                type_object="denoiser",
                tiling_dims=(-3, -2, -1),
                patch_size=self.patch_size,
                overlap=self.overlap,
                max_batch_size=self.max_batch_size,
                checkpoint_batches=self.checkpoint_batches,
            )
        return model, module, info

    def _amp(self, model):
        """``model``, autocasting each call and returning fp32.

        Wrapping once is what keeps the denoiser passes from drifting apart:
        besides the two in the step, ``eq_loss`` and ``_score`` call the model
        directly, and in demo_cyo those needed a second autocast site of their
        own (``trainer._amp_model``). Here there is one wrapped callable and no
        site left to miss.

        The ``.float()`` is load-bearing, not cosmetic: astra asserts float32
        input, and ``GpuFSC``'s ``torch.fft`` has no autocast policy — it raises
        outright on bfloat16. Only the denoiser runs reduced-precision; the
        physics and the losses always see fp32.

        A closure, not an ``nn.Module``: ``self.module`` already holds the
        parameters for the optimiser, the gradient clip and the train/eval
        switch, so nothing downstream needs this to be a module.
        """
        if self._amp_dtype is None:
            return model  # strict no-op on fp32

        def _call(x):
            with torch.amp.autocast(self._amp_device, dtype=self._amp_dtype):
                return model(x).float()

        return _call

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, cb):
        # Both tiling the denoiser and sharding the physics need a context:
        # the sharded operator asks it which shards this rank holds.
        if (self.distribute_model or self._wants_sharding()) and self.ctx is None:
            self.ctx = DistributedContext()

        pair, y_evn, y_odd = self._setup_physics()
        model, module, info = self._setup_model()
        self.module = module
        # Every model call below — step, eq_loss, _score — goes through this.
        model = self._amp(model)
        optimizer = torch.optim.Adam(module.parameters(), lr=float(self.learning_rate))
        transform = Rotate3D(volume_shape=pair.volume_shape)
        self._fsc = GpuFSC(device=self.device)
        print(
            f"Components set up: {info}, backend={pair.backend}, "
            f"num_operators={pair.num_operators}."
        )

        # Untrained reconstruction, so benchopt's evaluation at step 0 has a
        # valid result — the same contract the unrolled solver follows.
        module.eval()
        with torch.no_grad():
            self._score(model, pair, model(pair.init_evn), model(pair.init_odd))

        print("Starting tomo_ei training (one step per iteration).")
        module.train()
        for _ in distributed_callback_iter(
            cb, self.distributed_mode, self.device, self.ctx
        ):
            self._step(model, optimizer, transform, pair, y_evn, y_odd)
            self.profiler.end_iteration(self.ctx)
        sync_and_barrier(self.device, self.ctx)

    def _wants_sharding(self) -> bool:
        """Whether ``num_operators`` asks for a sharded operator at all.

        Accepts the config spellings of "unsharded" — ``None`` and the strings
        a YAML file or CLI hands through.
        """
        if self.num_operators is None:
            return False
        if isinstance(self.num_operators, str):
            return self.num_operators.strip().lower() not in ("none", "null")
        return True

    def _step(self, model, optimizer, transform, pair, y_evn, y_odd):
        optimizer.zero_grad(set_to_none=True)

        with self.profiler.track_step("forward"):
            # model is the _amp wrapper: it autocasts the denoiser and hands
            # back fp32, so the physics and the losses always see fp32.
            x_net = model(pair.init_evn)
            y_net = model(pair.init_odd)
            loss = obs_loss(pair, x_net, y_net, y_evn, y_odd)
            if self.eq_weight > 0:
                loss = loss + self.eq_weight * eq_loss(
                    pair, model, transform, x_net, y_net
                )

        with self.profiler.track_step("backward"):
            (self._scaler.scale(loss) if self._scaler is not None else loss).backward()

        if self._scaler is not None:
            # Clip the true gradients, not the scaled ones.
            self._scaler.unscale_(optimizer)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), float(self.grad_clip)
            )
        if self._scaler is None:
            optimizer.step()
        else:
            prev_scale = self._scaler.get_scale()
            self._scaler.step(optimizer)
            self._scaler.update()
            # On overflow scaler.step() silently drops the update and halves the
            # scale. A dropped step costs the same time as a taken one, so the
            # fp16 timings only mean something read next to this counter.
            self.amp_skipped += prev_scale > self._scaler.get_scale()
        self.loss = loss.item()

        # The metric is timed like any other region, and the region covers the
        # whole of it: the two-pass score costs another A + fbp + denoiser per
        # half, which is most of what "computing the metric" now means.
        with self.profiler.track_step("fsc"):
            with torch.no_grad():
                self._score(model, pair, x_net.detach(), y_net.detach())

    def _score(self, model, pair, x_net, y_net):
        """FSC between the two half-set reconstructions, scored two-pass.

        Each half is sent back through its own geometry and reconstructed
        before scoring — ``f(fbp(A(f(.))))``, demo_cyo's ``half_set_recon``.
        The round trip is what imprints the real missing-angle pattern on the
        reconstruction, and it is what makes the number comparable to the
        demo's. The halves are scored apart, never pre-averaged: FSC measures
        the agreement between two independent half-sets.

        No special case for sharded physics — ``A`` returns one measurement per
        shard and ``ShardedTomography.fbp`` consumes exactly that.
        """
        r_evn = model(pair.physics_evn.fbp(pair.physics_evn.A(x_net)))
        r_odd = model(pair.physics_odd.fbp(pair.physics_odd.A(y_net)))
        curve = self._fsc(r_evn, r_odd)
        shell, resolution, _ = fsc_resolution(
            curve, r_evn.squeeze().shape, self.pixel_size, self.fsc_threshold
        )
        self.fsc_shell, self.fsc_res = int(shell), float(resolution)
        self.reconstruction = 0.5 * (r_evn + r_odd)

    def get_result(self):
        result = dict(
            reconstruction=self.reconstruction,
            fsc_res=self.fsc_res,
            fsc_shell=self.fsc_shell,
            train_loss=self.loss,
            # Both None off the fp16 path; the objective drops None values, so
            # these become columns only where they mean something.
            amp_scale=self._scaler.get_scale() if self._scaler is not None else None,
            amp_skipped=self.amp_skipped if self._scaler is not None else None,
        )
        if self.profiler is not None:
            result.update(self.profiler.get_current_metrics())
        return result
