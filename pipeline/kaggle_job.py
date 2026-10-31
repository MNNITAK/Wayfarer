"""Runs ON Kaggle (pushed by kaggle_train.py): train splatfacto on the attached nerfstudio dataset, export the splat.

Settings are prepended by kaggle_train.py as plain assignments (ITERS = ...). Results land in /kaggle/working:
splat.ply, config.yml, dataparser_transforms.json, transforms.json, metrics.json.
"""
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ITERS = globals().get('ITERS', 30000)
WORK, TMP = Path('/kaggle/working'), Path('/kaggle/temp')
t0 = time.time()


def run(*cmd):
    print(f'\n[{(time.time() - t0) / 60:5.1f} min] $ {" ".join(map(str, cmd))}', flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


run('nvidia-smi', '--query-gpu=name,memory.total', '--format=csv')
# gsplat compiles its CUDA kernels on first use; build only for this GPU
os.environ['TORCH_CUDA_ARCH_LIST'] = subprocess.run(
    [sys.executable, '-c', 'import torch; print("%d.%d" % torch.cuda.get_device_capability())'],
    capture_output=True, text=True, check=True).stdout.strip()
run(sys.executable, '-m', 'pip', 'install', '-q', 'nerfstudio', 'ninja')

# the dataset: transforms.json + images/ (Kaggle unpacks zipped folders, but handle a leftover images.zip too)
src = Path(glob.glob('/kaggle/input/**/transforms.json', recursive=True)[0]).parent
data = TMP / 'data'
shutil.copytree(src, data, dirs_exist_ok=True)
for z in data.glob('*.zip'):
    with zipfile.ZipFile(z) as f:
        nested = all(n.startswith(f'{z.stem}/') for n in f.namelist())  # zip of the folder vs of its contents
        f.extractall(data if nested else data / z.stem)
    z.unlink()
print('images:', len(list((data / 'images').glob('*'))), flush=True)

out = TMP / 'outputs'
run('ns-train', 'splatfacto', '--data', data, '--output-dir', out, '--max-num-iterations', ITERS,
    '--viewer.quit-on-train-completion', 'True', '--vis', 'tensorboard',
    '--pipeline.datamanager.cache-images', 'cpu', '--pipeline.datamanager.cache-images-type', 'uint8',
    '--pipeline.model.use-bilateral-grid', 'True',  # same flags as wayfarer.py
    '--pipeline.model.use-scale-regularization', 'True',
    '--pipeline.model.camera-optimizer.mode', 'SO3xR3',  # the scan's poses are off by a few cm; refine them
    'nerfstudio-data', '--downscale-factor', '1')
config = sorted(out.glob('**/config.yml'), key=os.path.getmtime)[-1]
run('ns-export', 'gaussian-splat', '--load-config', config, '--output-dir', TMP / 'export')
try:
    run('ns-eval', '--load-config', config, '--output-path', WORK / 'metrics.json')
except subprocess.CalledProcessError:
    print('ns-eval failed; the splat is still fine', flush=True)

shutil.copy(TMP / 'export' / 'splat.ply', WORK)
shutil.copy(config, WORK)
shutil.copy(config.parent / 'dataparser_transforms.json', WORK)
shutil.copy(data / 'transforms.json', WORK)
print(f'done in {(time.time() - t0) / 60:.0f} min', flush=True)
