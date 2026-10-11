"""Фізичний поділ AID на train / val / test з добором різноманітних val/test-зображень.

З кожного класу беремо 15 зображень у val і 20 у test так, щоб серед цих 35 не було
ідентичних або дуже схожих. Схожість міряємо косинусною близькістю ознак ResNet18 (ImageNet):
вони реагують і на майже-дублікати (сусідні/перекриті знімки), і на однакові за виглядом сцени.

Алгоритм для кожного класу:
  1. перемішуємо зображення (SEED) — щоб вибірка лишалась репрезентативною, а не складалась з викидів;
  2. проходимо по черзі і беремо зображення, лише якщо його близькість до всіх уже взятих < поріг;
  3. поріг — квантиль Q попарних близькостей у межах класу (тобто жодна пара серед 35 не входить
     у найсхожіші (1 - Q) пар класу, при Q = 0.80 — у верхні 20%); якщо 35 не набирається — поріг поступово послаблюємо.
Решта зображень класу йде в train. Файли розкладаються жорсткими посиланнями (без дублювання на диску).

Запуск:  .venv/bin/python make_split.py              # обчислити поділ заново
         .venv/bin/python make_split.py --from-csv   # лише розкласти файли за готовим artifacts/split.csv
"""
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

SEED = 42
N_VAL, N_TEST = 15, 20
QUANTILE = 0.80          # стартовий поріг схожості (квантиль попарних близькостей у класі)
SRC_DIR = Path('data/AID')
OUT_DIR = Path('data/AID_split')
EMB_FILE = Path('artifacts/embeddings.npz')
IMG_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


class ImgDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(SRC_DIR / self.paths[i]).convert('RGB'))


def compute_embeddings(paths):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    loader = DataLoader(ImgDataset(paths, weights.transforms()), batch_size=64, num_workers=4)
    feats = []
    with torch.inference_mode():
        for i, x in enumerate(loader):
            feats.append(model(x))
            print(f'\rембедінги: {min((i + 1) * 64, len(paths))}/{len(paths)}', end='', flush=True)
    print()
    feats = torch.cat(feats).numpy()
    return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def select_diverse(emb, n, rng):
    sim = emb @ emb.T
    pair_sims = sim[np.triu_indices(len(emb), k=1)]
    order = rng.permutation(len(emb))
    q = QUANTILE
    while True:
        thr = np.quantile(pair_sims, q)
        chosen = []
        for i in order:
            if not chosen or sim[i, chosen].max() < thr:
                chosen.append(i)
                if len(chosen) == n:
                    return chosen, thr, q
        q = min(q + 0.01, 1.0)


def main():
    classes = sorted(d.name for d in SRC_DIR.iterdir() if d.is_dir())
    paths = [str(f.relative_to(SRC_DIR)) for c in classes
             for f in sorted((SRC_DIR / c).iterdir()) if f.suffix.lower() in IMG_EXT]
    df = pd.DataFrame({'path': paths, 'class': [p.split('/')[0] for p in paths]})
    df['label'] = df['class'].map({c: i for i, c in enumerate(classes)})

    if EMB_FILE.exists() and list(np.load(EMB_FILE, allow_pickle=True)['paths']) == paths:
        emb = np.load(EMB_FILE)['emb']
    else:
        emb = compute_embeddings(paths)
        EMB_FILE.parent.mkdir(exist_ok=True)
        np.savez(EMB_FILE, emb=emb, paths=np.array(paths))

    rng = np.random.default_rng(SEED)
    df['split'] = 'train'
    report = []
    for c in classes:
        idx = np.where(df['class'] == c)[0]
        chosen, thr, q = select_diverse(emb[idx], N_VAL + N_TEST, rng)
        chosen = idx[chosen]
        rng.shuffle(chosen)
        df.loc[chosen[:N_VAL], 'split'] = 'val'
        df.loc[chosen[N_VAL:], 'split'] = 'test'
        s = emb[chosen] @ emb[chosen].T
        report.append((c, len(idx), q, thr, s[np.triu_indices(len(chosen), k=1)].max()))

    print(pd.DataFrame(report, columns=['class', 'n', 'quantile', 'threshold', 'max_sim_selected'])
          .round(3).to_string(index=False))

    df.to_csv('artifacts/split.csv', index=False)
    write_split(df)


def write_split(df):
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for p, split in zip(df['path'], df['split']):
        dst = OUT_DIR / split / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(SRC_DIR / p, dst)
        except OSError:
            shutil.copy2(SRC_DIR / p, dst)
    print(df['split'].value_counts().to_dict(), '->', OUT_DIR)


if __name__ == '__main__':
    if '--from-csv' in sys.argv:
        write_split(pd.read_csv('artifacts/split.csv'))
    else:
        random.seed(SEED)
        torch.manual_seed(SEED)
        main()
