"""Evaluate a checkpoint on the cross-dataset benchmark.

Runs the model on every dataset in ``eval_datasets`` of the config (nuScenes,
Waymo, Lyft and ONCE val) and reports mAP and NDS. The model is loaded once;
datasets are processed one after the other (frames in temporal order, as
required by the streaming memory).

Example:
    python tools/eval_cross_dataset.py \\
        projects/configs/magnet3d/magnet3d_r50_nuscenes.py \\
        ckpts/magnet3d_r50_nuscenes.pth --out-dir work_dirs/eval/magnet3d_nuscenes
"""
import argparse
import importlib
import json
import os
import os.path as osp
import sys

import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from mmdet3d.apis import single_gpu_test  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('config', help='model config')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-dir', default=None,
                        help='output directory (default: work_dirs/eval/<config name>)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='subset of eval_datasets to run (default: all)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='only use the first N frames of every dataset')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override settings in the config')
    return parser.parse_args()


def main():
    args = parse_args()
    # disable TF32 for reproducible predictions
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    out_dir = args.out_dir or osp.join(
        'work_dirs', 'eval', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(out_dir, exist_ok=True)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model = MMDataParallel(model, device_ids=[0])

    results = {}
    for name in args.datasets or list(cfg.eval_datasets.keys()):
        ds_cfg = cfg.eval_datasets[name].copy()
        if args.max_frames:
            ds_cfg.update(overfit=True, overfit_samples=args.max_frames)
        dataset = build_dataset(ds_cfg)
        loader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
            dist=False, shuffle=False,
            nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', dict(type='DistributedSampler')))
        # every dataset starts with an empty temporal memory
        model.module.prev_scene_token = None
        if getattr(model.module, 'pts_bbox_head', None) is not None:
            model.module.pts_bbox_head.reset_memory()

        print(f'\nEvaluating on {name} ({len(dataset)} frames)')
        outputs = single_gpu_test(model, loader)
        # dict with mAP, NDS (translation, scale and orientation errors, no velocity) and the errors
        metrics = dataset.evaluate(outputs, osp.join(out_dir, name),
                                   train_classes=cfg.get('class_names', None))
        metrics = next(iter(metrics.values()))
        results[name] = metrics
        print(f'{name}: mAP {100 * metrics["mAP"]:.1f}  NDS {100 * metrics["NDS"]:.1f}')

    with open(osp.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\n' + ' ' * 6 + ''.join(f'{n:>10s}' for n in results))
    for key, label in (('mAP', 'mAP'), ('NDS', 'NDS')):
        print(f'{label:6s}' + ''.join(f'{100 * m[key]:10.1f}' for m in results.values()))


if __name__ == '__main__':
    main()
