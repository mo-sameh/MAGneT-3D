# Data preparation

MAGneT-3D uses the front camera of nuScenes, Waymo, Lyft and ONCE in a common format: one COCO-style
annotation file per split with frames, videos, 3D boxes and the shared taxonomy
(`vehicle`, `pedestrian`, `bicycle`). The datasets are downloaded and converted with
[time-dataset-converter](https://gitlab.lrz.de/johannes-tum/time-dataset-converter); its
`prepare_data.sh` writes the layout below, so its `data/` folder can be used (or linked) as `data/`
here. The expected layout is

```
data/
  joint/
    nuscenes_train.json   nuscenes_val.json
    waymo_train.json      waymo_val.json
    ONCE_val.json
  lyft_filtered_train.json  lyft_filtered_val.json
  nuscenes/   waymo/   lyft/   ONCE/     # images, paths as in the annotation files
```

Image paths in the annotation files are relative to `data/`.
