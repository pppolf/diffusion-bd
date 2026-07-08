# Linux Run

上传项目后执行：

```bash
cd /home/a430/yh/diffusion-bd

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffusion-bd

chmod +x env_linux.sh
chmod +x prepare_linux_demo.sh
chmod +x train_smoke_poisoned.sh
chmod +x run_linux_demo.sh

./run_linux_demo.sh
```

如果 COCO 实际不在 `/home/a430/data/coco2017`，可以这样覆盖：

```bash
export COCO_ROOT="/实际的/coco2017/目录"
./run_linux_demo.sh
```
