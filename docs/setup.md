# Environment setup

Tested with Python 3.8, PyTorch 1.9.0 (CUDA 11.1), mmcv-full 1.6.0, mmdet 2.28.2 and
mmdetection3d v1.0.0rc6 (GPUs up to Ampere).

```bash
conda create -n magnet3d python=3.8 -y
conda activate magnet3d

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.6.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

git clone https://github.com/mo-sameh/MAGneT-3D.git
cd MAGneT-3D

# mmdetection3d is used unmodified
git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
pip install -e mmdetection3d

pip install -r requirements.txt

# metric of the cross-dataset benchmark
pip install git+https://github.com/mo-sameh/MAGneT-3D-Benchmark.git
```

[flash-attn](https://github.com/Dao-AILab/flash-attention) is optional and not needed for the
released models.
