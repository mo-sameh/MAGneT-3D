# Paper checkpoints

The checkpoints of Table 2 in the paper, with their evaluation configs

| Train | Config | Checkpoint | nuScenes | Waymo | Lyft | ONCE | Avg. cross |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| nuScenes | [config](configs/magnet3d_r50_nuscenes_table2.py) | [model](https://drive.google.com/file/d/1v5Rf2ktekq-g7GOH0FjzOzUfhTuDC94q/view?usp=sharing) | **28.5** / **32.9** | 10.1 / 17.2 | 14.3 / 19.5 | 12.4 / 19.1 | 12.3 / 18.6 |
| Waymo | [config](configs/magnet3d_r50_waymo_table2.py) | [model](https://drive.google.com/file/d/1Beawbpf03MUsn5XSJDtMIaW0od55klxU/view?usp=sharing) | 13.4 / 18.8 | **34.5** / **39.3** | 13.8 / 19.5 | 9.2 / 14.0 | 12.2 / 17.4 |
| Lyft | [config](configs/magnet3d_r50_lyft_table2.py) | [model](https://drive.google.com/file/d/1ajrv9Q6i8wC23azaPbh1zScB8Ggt3-uV/view?usp=sharing) | 4.0 / 7.7 | 5.2 / 11.6 | **32.5** / **34.5** | 6.0 / 11.9 | 5.0 / 10.4 |

Download the checkpoints into this folder and run from the repository root:

```bash
python tools/eval_cross_dataset.py table2/configs/magnet3d_r50_nuscenes_table2.py \
    table2/magnet3d_r50_nuscenes_table2.pth --out-dir work_dirs/eval/table2_nuscenes
```
