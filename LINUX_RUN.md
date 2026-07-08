# Run

The portable entry point is `run.py`. It lets you choose the platform workflow
explicitly from the command line:

```bash
python run.py linux run
python run.py linux prepare
python run.py linux train
```

```bat
python run.py windows run
python run.py windows prepare
python run.py windows train
```

Use `auto` to run the workflow for the current OS:

```bash
python run.py auto run
```

Common overrides:

```bash
COCO_ROOT=/actual/coco2017 python run.py linux run
python run.py windows train --set COCO_ROOT=D:\data\coco2017 --set BD_MAX_TRAIN_STEPS=3000
```

Linux still uses the original shell scripts:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffusion-bd
chmod +x env_linux.sh prepare_linux_demo.sh train_smoke_poisoned.sh run_linux_demo.sh
./run_linux_demo.sh
```

Windows experiment training uses `train_smoke_poisoned.bat` for compatibility
with the old filename, but its defaults are now experiment-oriented:

- output: `outputs\experiment_poisoned_windows`
- resolution: `512`
- max train steps: `2000`
- full `data_demo\poisoned\train` dataset, with no smoke sample cap
