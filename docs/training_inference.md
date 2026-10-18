# Training and evaluation

## Evaluation

Evaluate a checkpoint on the four validation sets:

```bash
python tools/eval_cross_dataset.py projects/configs/magnet3d/magnet3d_r50_nuscenes.py \
    ckpts/magnet3d_r50_nuscenes.pth --out-dir work_dirs/eval/magnet3d_nuscenes
```

Metrics are written to `<out-dir>/metrics.json`; `--datasets nuscenes waymo` evaluates a subset.
Evaluation runs on one GPU (frames are processed in temporal order).

## Training

MAGneT-3D is trained in two stages with the same config: a DRAG warm-up (`--drag-warmup`), then
the full model initialized from it (`--stage1-ckpt`).

```bash
# stage 1: DRAG warm-up (2 GPUs x 8 samples)
bash tools/dist_train.sh projects/configs/magnet3d/magnet3d_r50_nuscenes.py 2 \
    --work-dir work_dirs/magnet3d_r50_nuscenes_warmup --drag-warmup
# stage 2: full model
bash tools/dist_train.sh projects/configs/magnet3d/magnet3d_r50_nuscenes.py 2 \
    --work-dir work_dirs/magnet3d_r50_nuscenes \
    --stage1-ckpt work_dirs/magnet3d_r50_nuscenes_warmup/latest.pth
```

On a SLURM cluster use `tools/slurm_train.sh <partition> <job name> <config> <work dir> [args]`.
The configs assume 2 GPUs x 8 samples.
