# SafeFlow Rectified Flow

This directory is a non-invasive add-on for `dmpc_fm_cbf`. It reuses the
existing 5-channel model, canonical frame convention, torchdiffeq solver and
CBF projector. It does not modify the original FM trainer or checkpoint.

To install it in the SafeFlow project, copy this directory to:

```text
dmpc_fm_cbf/rectified/
```

The project root must be on `PYTHONPATH`:

```bash
cd /path/to/dmpc_fm_cbf
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Workflow

### 1. Generate a deterministic coupling

```bash
python -m rectified.reflow_generate \
  --checkpoint notebooks/cache/model_vel_5ch_canonical_r.pt \
  --data notebooks/cache/car_ring_track_oc_dataset_v1_v0rand_with_vel.npy \
  --output rectified/checkpoints/reflow_round1_pairs.npz \
  --cbf \
  --num-segments 8 \
  --ode-method rk4
```

The generated pair contains `x0`, `x1`, and `cond` with shapes `(N, 5, T)`,
`(N, 5, T)`, and `(N, 5)`. `x1` is produced by the current FM ODE, and the
same sampler can apply CBF projection when obstacle ellipses are supplied.

For real obstacle constraints, pass a JSON file with one list per sample:

```json
[
  [{"center": [3.0, 0.5], "radius": 1.2}],
  [],
  [{"center": [2.0, -0.8], "radius": 1.0}]
]
```

Without `--obstacles`, the CBF code path is still selected but there are no
obstacles to project against. This is appropriate for the local ring dataset,
which has no obstacle labels, but it is not an obstacle-safety claim.

### 2. Warm-start reflow training

```bash
python -m rectified.train_reflow \
  --pairs rectified/checkpoints/reflow_round1_pairs.npz \
  --init-checkpoint notebooks/cache/model_vel_5ch_canonical_r.pt \
  --output rectified/checkpoints/model_reflow_round1.pt \
  --epochs 100 \
  --batch-size 32 \
  --device cpu
```

The original checkpoint is used as a warm start. The output checkpoint is
separate and has the same model architecture. The trainer uses the standard
affine interpolation but trains on the deterministic `(x0, x1)` pairs.

### 3. NFE/latency evaluation

```bash
python -m rectified.eval_nfe \
  --checkpoint rectified/checkpoints/model_reflow_round1.pt \
  --data notebooks/cache/car_ring_track_oc_dataset_v1_v0rand_with_vel.npy \
  --steps 1 2 4 8 16 \
  --device cpu
```

This reports endpoint error against the reference data and mean wall-clock
sampling time. It is an offline comparison; it does not replace the NuPlan
closed-loop evaluation.

## Design constraints

- The original FM files and original checkpoint are untouched.
- CBF is used during pair generation, not as a hidden modification to the
  training loss.
- The default coupling generation is deterministic (`seed + sample index`).
- A generated pair is not automatically a valid road trajectory: CBF protects
  supplied obstacle ellipses, while road-boundary validation still belongs to
  the downstream map-aware planner.
