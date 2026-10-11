"""Дашборд метрик для кожного запуску (exp3–exp7, студенти 200 епох).

Для кожного запуску зберігає plots/<run>.png:
  - криві навчання (loss, accuracy, val F1) з позначкою найкращої епохи;
  - підсумкові accuracy / macro-F1 на train / val / test;
  - стійкість до спотворень (якщо вимірювалась);
  - F1 по класах на тесті;
  - матриця плутанини на тесті.
Плюс plots/summary.png — порівняння всіх запусків.

Запуск: .venv/bin/python plot_runs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

ROOT = Path(__file__).parent
OUT = ROOT / "plots"
CLASSES = sorted(p.name for p in (ROOT / "data/AID").iterdir() if p.is_dir())

# палітра: категоріальні слоти 1–3, нейтральні чорнила, послідовна синя рампа
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2, "font.size": 10,
    "lines.linewidth": 2,
})

# (назва запуску, ключ в історії/предиктах, history json, preds npz, results csv, ключ у results)
RUNS = [
    ("exp3_frozen", "exp3_frozen", "steps6_10/out/histories_steps6_10.json", "steps6_10/out/test_preds_all.npz"),
    ("exp4_frozen_aug", "exp4_frozen_aug", "steps6_10/out/histories_steps6_10.json", "steps6_10/out/test_preds_all.npz"),
    ("exp5_finetune", "exp5_finetune", "steps6_10/out/histories_steps6_10.json", "steps6_10/out/test_preds_all.npz"),
    ("exp6_student", "exp6_student", "steps6_10/out/histories_steps6_10.json", "steps6_10/out/test_preds_all.npz"),
    ("exp7_student_kd", "exp7_student_kd", "steps6_10/out/histories_steps6_10.json", "steps6_10/out/test_preds_all.npz"),
    ("student_200ep", "student", "students200/out/histories_students_200ep.json", "students200/out/test_preds_students_200ep.npz"),
    ("student_kd_200ep", "student_kd", "students200/out/histories_students_200ep.json", "students200/out/test_preds_students_200ep.npz"),
]

RESULTS = pd.concat([
    pd.read_csv(ROOT / "steps6_10/out/results_all.csv", index_col=0),
    pd.read_csv(ROOT / "steps3_4/out/results_steps3_4.csv", index_col=0)[["train_acc", "train_f1"]]
      .reindex(["exp3_frozen", "exp4_frozen_aug"]),
], axis=1)
# train_acc/train_f1 для exp5–exp7 беремо з інших таблиць
_r5 = pd.read_csv(ROOT / "step5/out/results_step5.csv", index_col=0)
RESULTS.loc["exp5_finetune", ["train_acc", "train_f1"]] = _r5.loc["exp5_finetune", ["train_acc", "train_f1"]]
_r200 = pd.read_csv(ROOT / "students200/out/results_students_200ep.csv", index_col=0)
ROBUST = pd.read_csv(ROOT / "steps6_10/out/robustness.csv", index_col=0)


def summary_row(run, key, hist, y, p):
    """train/val/test acc і F1 для запуску. train для студентів 40 еп. — з останньої епохи кривої."""
    if run.endswith("_200ep"):
        r = _r200.loc[key]
        train_acc, train_f1 = r["train_acc"], np.nan
        val_acc, val_f1 = r["val_acc"], r["val_f1"]
    else:
        r = RESULTS.loc[run]
        train_acc, train_f1 = r.get("train_acc", np.nan), r.get("train_f1", np.nan)
        val_acc, val_f1 = r["val acc"], r["val F1"]
    if np.isnan(train_acc):
        train_acc = hist["train_acc"][hist["best_epoch"] - 1]
    return {
        "acc": [train_acc, val_acc, accuracy_score(y, p)],
        "f1": [train_f1, val_f1, f1_score(y, p, average="macro")],
    }


def plot_curves(axes, hist):
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    best = hist["best_epoch"]
    specs = [
        ("Loss", [("train", hist["train_loss"], BLUE), ("val", hist["val_loss"], ORANGE)]),
        ("Accuracy", [("train", hist["train_acc"], BLUE), ("val", hist["val_acc"], ORANGE)]),
        ("Val macro-F1", [("val", hist["val_f1"], ORANGE)]),
    ]
    for ax, (title, series) in zip(axes, specs):
        for label, vals, c in series:
            ax.plot(ep, vals, color=c, label=label)
        ax.axvline(best, color=MUTED, lw=1, ls="--")
        ax.set_title(title)
        ax.set_xlabel("епоха")
        if title != "Loss":
            ax.set_ylim(0, 1.02)
            v = series[-1][1][best - 1]
            ax.plot(best, v, "o", ms=8, color=series[-1][2], mec=SURFACE, mew=2, zorder=5)
            ax.annotate(f"best ep {best}: {v:.3f}", (best, v), xytext=(-10, 10),
                        textcoords="offset points", ha="right", va="bottom", color=INK2, fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
        if len(series) > 1:
            ax.legend(loc="best")


def plot_summary(ax, s):
    x = np.arange(3)
    w = 0.36
    for i, (name, vals, c) in enumerate([("accuracy", s["acc"], BLUE), ("macro-F1", s["f1"], AQUA)]):
        xs = x + (i - 0.5) * w
        vals = np.array(vals, dtype=float)
        ax.bar(xs, np.nan_to_num(vals), w - 0.04, color=c, label=name)
        for xi, v in zip(xs, vals):
            ax.text(xi, (0 if np.isnan(v) else v) + 0.015, "н/д" if np.isnan(v) else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, ["train", "val", "test"])
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_title("Підсумкові метрики")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.1))


def plot_robust(ax, run):
    if run not in ROBUST.index:
        ax.axis("off")
        ax.text(0.5, 0.5, "Стійкість до спотворень\nдля цього запуску не вимірювалась",
                ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return
    r = ROBUST.loc[run]
    cols = [c for c in ROBUST.columns if c not in ("середня (спотворені)", "середнє падіння")]
    vals = r[cols].values.astype(float)
    y = np.arange(len(cols))[::-1]
    colors = [MUTED if c == "чистий" else BLUE for c in cols]
    ax.barh(y, vals, 0.7, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, cols)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Стійкість: test acc (сер. падіння {r['середнє падіння']:.3f})")


def plot_per_class(ax, y, p):
    f1 = f1_score(y, p, average=None, labels=range(len(CLASSES)))
    order = np.argsort(f1)
    yy = np.arange(len(CLASSES))
    ax.barh(yy, f1[order], 0.72, color=BLUE)
    for yi, v in zip(yy, f1[order]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.axvline(f1.mean(), color=ORANGE, lw=1.5, ls="--")
    ax.text(f1.mean(), len(CLASSES) - 0.2, f" macro {f1.mean():.3f}", color=INK2, fontsize=8.5, va="bottom")
    ax.set_yticks(yy, [CLASSES[i] for i in order], fontsize=8.5)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.6, len(CLASSES) + 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_title("Test F1 по класах")


def plot_cm(ax, fig, y, p):
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
    im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=cm.max())
    n = len(CLASSES)
    ax.set_xticks(range(n), CLASSES, rotation=90, fontsize=7)
    ax.set_yticks(range(n), CLASSES, fontsize=7)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="#ffffff" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_xlabel("передбачений клас")
    ax.set_ylabel("справжній клас")
    ax.set_title("Матриця плутанини (test)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def plot_run(run, key, hist_path, preds_path):
    hist = json.loads((ROOT / hist_path).read_text())[key]
    d = np.load(ROOT / preds_path)
    y, p = d["y_test"], d[key]
    s = summary_row(run, key, hist, y, p)

    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.15, 1.15], width_ratios=[1, 1, 1.55],
                          hspace=0.38, wspace=0.28)
    plot_curves([fig.add_subplot(gs[0, i]) for i in range(3)], hist)
    plot_summary(fig.add_subplot(gs[1, 0]), s)
    plot_robust(fig.add_subplot(gs[2, 0]), run)
    plot_per_class(fig.add_subplot(gs[1:, 1]), y, p)
    plot_cm(fig.add_subplot(gs[1:, 2]), fig, y, p)

    extra = f"  ·  train time {hist['train_time'] / 60:.1f} хв"
    if "T" in hist:
        extra += f"  ·  KD T={hist['T']:g}, α={hist['alpha']:g}"
    fig.suptitle(f"{run}  —  test acc {s['acc'][2]:.3f}, test macro-F1 {s['f1'][2]:.3f}{extra}",
                 x=0.06, y=0.94, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.savefig(OUT / f"{run}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return s


def plot_overview(summaries):
    runs = list(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x = np.arange(len(runs))
    w = 0.38
    for ax, metric, title in [(axes[0], "acc", "Accuracy"), (axes[1], "f1", "Macro-F1")]:
        for i, (split, c) in enumerate([("val", ORANGE), ("test", BLUE)]):
            idx = 1 if split == "val" else 2
            vals = [summaries[r][metric][idx] for r in runs]
            xs = x + (i - 0.5) * w
            ax.bar(xs, vals, w - 0.04, color=c, label=split)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, runs, rotation=25, ha="right")
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
        ax.set_ylim(0, 1.15)
    fig.suptitle("Порівняння запусків", x=0.06, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    summaries = {}
    for run, *args in RUNS:
        summaries[run] = plot_run(run, *args)
        print(f"plots/{run}.png  test acc={summaries[run]['acc'][2]:.4f}  test F1={summaries[run]['f1'][2]:.4f}")
    plot_overview(summaries)
    print("plots/summary.png")
