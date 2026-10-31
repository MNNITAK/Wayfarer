"""Train the splat on a free Kaggle GPU instead of a local one (e.g. from a Mac).

  python pipeline/kaggle_train.py push   --data work/apartment_ns --name apartment   # upload + start training
  python pipeline/kaggle_train.py status --name apartment
  python pipeline/kaggle_train.py pull   --name apartment --title Apartment --metric  # download -> viewer scene

push uploads the nerfstudio dataset (from rgbd_to_nerfstudio.py, or work/<name>/ns from wayfarer.py) as a private
Kaggle dataset and runs kaggle_job.py on a T4 with the same splatfacto flags as wayfarer.py. pull downloads the
trained splat and runs navgrid.py on it, like the last step of wayfarer.py.

Needs the Kaggle CLI (pip install kaggle) and a token in ~/.kaggle/access_token. Kaggle notebooks need a
phone-verified account for GPU and internet (the job pip-installs nerfstudio).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = 'kaggle'


def kaggle(*args, capture=False):
    r = subprocess.run([KAGGLE, *map(str, args)], text=True, capture_output=capture)
    if r.returncode:
        sys.exit(f'kaggle {" ".join(map(str, args))} failed' + (f':\n{r.stdout}{r.stderr}' if capture else ''))
    return r.stdout if capture else None


def username():
    m = re.search(r'username:\s*(\S+)', kaggle('config', 'view', capture=True))
    if not m or m.group(1) == 'None':
        sys.exit('no Kaggle user configured: save your token to ~/.kaggle/access_token')
    return m.group(1)


def ids(name):
    user = username()
    slug = re.sub(r'[^a-z0-9-]+', '-', name.lower()).strip('-')
    return f'{user}/wayfarer-{slug}-data', f'{user}/wayfarer-{slug}-train'


def push(a):
    data_id, kernel_id = ids(a.name)
    if not (a.data / 'transforms.json').exists():
        sys.exit(f'{a.data} has no transforms.json')
    (a.data / 'dataset-metadata.json').write_text(json.dumps(
        {'title': data_id.split('/')[1], 'id': data_id, 'licenses': [{'name': 'CC0-1.0'}]}))
    if not a.no_upload:
        exists = subprocess.run([KAGGLE, 'datasets', 'status', data_id], capture_output=True).returncode == 0
        print(f'uploading {a.data} to the private dataset {data_id} ...')
        if exists:
            kaggle('datasets', 'version', '-p', a.data, '-m', 'update', '--dir-mode', 'zip')
        else:
            kaggle('datasets', 'create', '-p', a.data, '--dir-mode', 'zip')
    for _ in range(90):  # a new dataset answers 403 for a while until Kaggle has processed it
        r = subprocess.run([KAGGLE, 'datasets', 'status', data_id], capture_output=True, text=True)
        if 'ready' in r.stdout:
            break
        time.sleep(10)
    else:
        sys.exit(f'dataset {data_id} is not ready after 15 min; check it on kaggle.com, then push with --no-upload')

    kdir = a.work / 'kaggle' / a.name
    kdir.mkdir(parents=True, exist_ok=True)
    job = (Path(__file__).parent / 'kaggle_job.py').read_text()
    (kdir / 'kaggle_job.py').write_text(f'ITERS = {a.iters}\n' + job)
    (kdir / 'kernel-metadata.json').write_text(json.dumps({
        'id': kernel_id, 'title': kernel_id.split('/')[1], 'code_file': 'kaggle_job.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': [data_id], 'competition_sources': [], 'kernel_sources': []}, indent=1))
    kaggle('kernels', 'push', '-p', kdir, '--accelerator', 'NvidiaTeslaT4')
    print(f'training started: https://www.kaggle.com/code/{kernel_id}\n'
          f'check with: python pipeline/kaggle_train.py status --name {a.name}')


def status(a):
    print(kaggle('kernels', 'status', ids(a.name)[1], capture=True).strip())


def pull(a):
    kernel_id = ids(a.name)[1]
    out = a.work / 'kaggle' / a.name / 'output'
    out.mkdir(parents=True, exist_ok=True)
    kaggle('kernels', 'output', kernel_id, '-p', out, '-o')
    if not (out / 'splat.ply').exists():
        log = next(out.glob('*.log'), None)
        sys.exit(f'no splat.ply in the job output yet; see {log or out}')
    cmd = [sys.executable, Path(__file__).parent / 'navgrid.py', out / 'splat.ply',
           '--out', ROOT / 'web' / 'public' / 'scenes' / a.name, '--name', a.title or a.name,
           '--transforms', out / 'transforms.json', '--dataparser', out / 'dataparser_transforms.json']
    if a.metric:  # poses were in metres (RGB-D scan): a splat unit is 1/scale metres
        cmd += ['--scale', 1 / json.loads((out / 'dataparser_transforms.json').read_text())['scale']]
    subprocess.run([str(c) for c in cmd], check=True)
    if (out / 'metrics.json').exists():
        print('held-out views:', json.loads((out / 'metrics.json').read_text()).get('results'))


def main():
    global KAGGLE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['push', 'status', 'pull'])
    ap.add_argument('--name', required=True, help='scene id: web/public/scenes/<name>')
    ap.add_argument('--data', type=Path, help='push: nerfstudio dataset folder (transforms.json + images/)')
    ap.add_argument('--no-upload', action='store_true', help='push: the dataset is already on Kaggle')
    ap.add_argument('--iters', type=int, default=30000, help='push: training iterations')
    ap.add_argument('--title', help='pull: display name')
    ap.add_argument('--metric', action='store_true', help='pull: the poses were already in metres')
    ap.add_argument('--work', type=Path, default=ROOT / 'work')
    ap.add_argument('--kaggle', default='kaggle', help='path to the kaggle CLI')
    a = ap.parse_args()
    KAGGLE = a.kaggle
    if a.command == 'push' and not a.data:
        ap.error('push needs --data')
    {'push': push, 'status': status, 'pull': pull}[a.command](a)


if __name__ == '__main__':
    main()
