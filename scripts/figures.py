"""Regenerate the pitch figures from the saved evaluation artefacts.

Every number comes from files the evaluation scripts write, so the figures cannot drift from the
report: docs/figures/{detector_errors,roc_regions,auc_by_type,rotation_approaches}.png

Usage: python scripts/figures.py   (after train_keypoints.py, eval_pipeline.py, rotation_experiments.py)
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

OUT = Path("docs/figures")
# какие результаты рисуем: суффиксы файлов pipeline_eval / pipeline_oof и kp_oof_<region>
EVAL = __import__("os").environ.get("DXAQC_FIG_EVAL", "_axis24")
KP = __import__("os").environ.get("DXAQC_FIG_KP", "_ens3")
BLUE, ORANGE, GREY = "#4C7BD9", "#E8833A", "#9AA0A6"
NAMES = {"col_top": "Столб, верх", "col_bottom": "Столб, уровень таза", "crest_a": "Крыло таза слева",
         "crest_b": "Крыло таза справа", "fh_c": "Центр головки", "fh_top": "Верх головки",
         "fn_c": "Шейка", "gt_top": "Большой вертел, верх", "gt_lat": "Большой вертел, край",
         "lt": "Малый вертел", "isch": "Седалищная кость", "shaft_p": "Диафиз под МВ",
         "shaft_d": "Диафиз внизу"}
TYPE_RU = {"spine:v_axis": "Наклон оси", "spine:v_pos": "Укладка (гребни)", "spine:v_artifact": "Инородные тела",
           "hip:v_roi": "Поля кадра бедра", "hip:v_posrot": "Ротация бедра"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)


def detector_errors():
    """Landmark error in mm, per point, from out-of-fold predictions vs the manual annotation."""
    idx = pd.read_csv("data/train/image_index.csv")
    idx["id"] = idx.n.map("{:03d}".format) + "_" + idx.region
    idx = idx.set_index("id")
    kp = json.loads(Path("data/train/keypoints.json").read_text())
    oof = {**json.loads(Path(f"data/train/kp_oof_spine{KP}.json").read_text()),
           **json.loads(Path(f"data/train/kp_oof_hip{KP}.json").read_text())}
    rows = []
    for i, d in oof.items():
        mm = float(idx.loc[i, "mm_per_px_y"])       # pixel size of that image (DICOM tag)
        for k, p in d["points"].items():
            q = kp[i]["points"].get(k)
            if p and q:
                rows.append({"point": k, "region": "spine" if i.endswith("spine") else "hip",
                             "err": float(np.hypot(p[0] - q[0], p[1] - q[1]) * mm)})
    e = pd.DataFrame(rows).groupby(["region", "point"]).err.agg(["median", lambda s: s.quantile(0.9)])
    e.columns = ["median", "p90"]
    e = e.reset_index().sort_values(["region", "median"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(e))
    for k, (_, r) in enumerate(e.iterrows()):
        c = BLUE if r.region == "spine" else ORANGE
        ax.plot([r["median"], r["p90"]], [y[k], y[k]], color=c, alpha=0.45, lw=2.5, solid_capstyle="round")
        ax.plot(r["median"], y[k], "o", color=c, ms=7)
        ax.text(r["p90"] + 0.3, y[k], f"{r['median']:.1f} мм", va="center", fontsize=9, color="#444")
    ax.set_yticks(y, [NAMES.get(p, p) for p in e.point])
    ax.set_xlabel("Ошибка, мм: точка — медиана, линия — до 90-го перцентиля")
    ax.set_title("Точность детектора ключевых точек")
    ax.plot([], [], "o-", color=BLUE, label="Позвоночник")
    ax.plot([], [], "o-", color=ORANGE, label="Бедро")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "detector_errors.png", dpi=150)
    plt.close(fig)


def roc_regions():
    """ROC of the image verdict score, per region, from the honest out-of-fold run."""
    df = pd.read_csv(f"data/train/pipeline_oof{EVAL}.csv")
    rep = json.loads(Path(f"data/train/pipeline_eval{EVAL}.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for name, sub, c in (("Позвоночник", df[df.id.str.endswith("spine")], BLUE),
                         ("Бёдра", df[df.id.str.contains("hip")], ORANGE),
                         ("Всего", df, "#3A3A3A")):
        fpr, tpr, _ = roc_curve(sub.y.astype(int), sub.score)
        key = {"Позвоночник": "spine", "Бёдра": "hip", "Всего": "overall"}[name]
        auc, ci = rep["regions"][key]["auc"]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name}: AUC {auc:.2f} [{ci[0]:.2f}–{ci[1]:.2f}]")
        sens, spec = rep["regions"][key]["sens"][0], rep["regions"][key]["spec"][0]
        ax.plot(1 - spec, sens, "o", color=c, ms=9, mfc="white", mew=2)   # shipped operating point
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    ax.set_xlabel("Доля ложных тревог (1 − специфичность)")
    ax.set_ylabel("Чувствительность")
    ax.set_title("Вердикт «есть нарушение качества»\n(кросс-валидация по пациентам)")
    ax.plot([], [], "o", color=GREY, mfc="white", mew=2, label="рабочая точка (screening)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style(ax)
    ax.grid(color="#E6E6E6", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "roc_regions.png", dpi=150)
    plt.close(fig)


def auc_by_type():
    rep = json.loads(Path(f"data/train/pipeline_eval{EVAL}.json").read_text())["types"]
    items = sorted(rep.items(), key=lambda kv: kv[1]["auc"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(items))
    for k, (name, v) in enumerate(items):
        c = BLUE if name.startswith("spine") else ORANGE
        ax.barh(y[k], v["auc"], color=c, height=0.55)
        ax.text(v["auc"] + 0.01, y[k], f"{v['auc']:.2f}  ({v['n_pos']} нарушений)", va="center", fontsize=9, color="#444")
    ax.axvline(0.5, color=GREY, lw=1, ls="--")
    ax.set_yticks(y, [TYPE_RU.get(n, n) for n, _ in items])
    ax.set_xlim(0.4, 1.12)
    ax.set_xlabel("ROC-AUC по типу нарушения")
    ax.set_title("Точность по типам нарушений")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "auc_by_type.png", dpi=150)
    plt.close(fig)


def rotation_approaches():
    """Every approach tried for hip rotation, as measured honestly (detector crops)."""
    vals = [("Геометрия по точкам", 0.72), ("HOG + лог. регрессия", 0.67), ("Своя CNN", 0.73),
            ("Многозадачная сеть\n(точки + ротация)", 0.69), ("HOG + случайный лес", 0.81)]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(vals))
    for k, (name, v) in enumerate(vals):
        c = ORANGE if k == len(vals) - 1 else GREY
        ax.barh(y[k], v, color=c, height=0.55)
        ax.text(v + 0.005, y[k], f"{v:.2f}", va="center", fontsize=9, color="#444")
    ax.axvline(0.5, color=GREY, lw=1, ls="--")
    ax.set_yticks(y, [n for n, _ in vals])
    ax.set_xlim(0.45, 0.9)
    ax.set_xlabel("ROC-AUC (кросс-валидация по пациентам, точки от детектора)")
    ax.set_title("Ротация бедра: что пробовали")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "rotation_approaches.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    detector_errors()
    roc_regions()
    auc_by_type()
    rotation_approaches()
    print("figures ->", OUT)
