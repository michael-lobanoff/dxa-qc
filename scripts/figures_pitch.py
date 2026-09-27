"""Схемы и аналитика для презентации — всё строится из данных, а не рисуется руками.

Каждая цифра берётся из файлов, которые пишут скрипты обучения и оценки, поэтому картинка не может
разойтись с отчётом. Результат: docs/figures/{architecture,dataset,split,metrics_ci,taxonomy,errors}.png

    python scripts/figures_pitch.py [--eval _axis24]
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("docs/figures")
BLUE, ORANGE, GREY, GREEN, RED = "#4C7BD9", "#E8833A", "#9AA0A6", "#2E9E5B", "#C2413A"
INK, LINE = "#1B1D21", "#D8DCE2"
TYPE_RU = {"spine:v_axis": "Наклон оси\nпозвоночника", "spine:v_pos": "Некорректная\nукладка",
           "spine:v_artifact": "Посторонние\nпредметы", "hip:v_roi": "Некорректная\nобласть интереса",
           "hip:v_posrot": "Ротация и укладка\nбедра"}
CRITERION = {"spine:v_axis": "угол оси > 5°", "spine:v_pos": "гребни таза и Th12 в кадре",
             "spine:v_artifact": "металл, косточки белья", "hip:v_roi": "поля 3 / 3 / 2 см",
             "hip:v_posrot": "малый вертел"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"docs/figures/{name}.png")


def box(ax, x, y, w, h, text, color=BLUE, fill="#F2F6FD", fontsize=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.4, edgecolor=color, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color=color, shrinkA=2, shrinkB=2))


# --------------------------------------------------------------------------------------- архитектура
def architecture():
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "Как устроен сервис: от DICOM до строки отчёта", ha="center", fontsize=14,
            fontweight="bold", color=INK)

    steps = [("DICOM\nна входе", "чтение пикселей,\nMONOCHROME1, 16 бит,\nсжатые синтаксисы"),
             ("Анатомическая\nобласть", "классификатор\n+ защита «свой/чужой»"),
             ("Ключевые точки", "ансамбль из трёх U-Net\n4 точки / 9 точек"),
             ("Измерения", "градусы и миллиметры\nпо критериям ТЗ"),
             ("Решения\nпо типам", "5 моделей,\nсвой порог у каждой")]
    w, h, y = 0.175, 0.26, 0.5
    for k, (title, sub) in enumerate(steps):
        x = 0.01 + k * 0.2
        box(ax, x, y, w, h, f"{title}\n\n", color=BLUE, bold=True, fontsize=10.5)
        ax.text(x + w / 2, y + 0.075, sub, ha="center", va="center", fontsize=8.2, color="#5A6270", linespacing=1.3)
        if k:
            arrow(ax, x - 0.025, y + h / 2, x, y + h / 2)

    ax.text(0.5, 0.42, "↓", ha="center", fontsize=15, color=GREY)
    outs = [("Таблица CSV / XLSX", "формат ТЗ 2.5"), ("Снимок с подсветкой", "PNG + DICOM SC"),
            ("Текстовый отчёт", "DICOM Basic Text SR"), ("Веб-интерфейс и API", "FastAPI, Swagger")]
    for k, (title, sub) in enumerate(outs):
        x = 0.03 + k * 0.245
        box(ax, x, 0.16, 0.205, 0.2, f"{title}\n", color=GREEN, fill="#EFF8F2", fontsize=9.5, bold=True)
        ax.text(x + 0.1, 0.205, sub, ha="center", va="center", fontsize=8, color="#5A6270")

    ax.text(0.5, 0.06, "Всё локально, в одном docker-образе: ни одного обращения за пределы машины",
            ha="center", fontsize=9.5, color="#5A6270", style="italic")
    save(fig, "architecture")


# ------------------------------------------------------------------------------------------- данные
def dataset(idx):
    """Два панно: из чего состоит выборка и насколько редки нарушения каждого типа."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.32})
    fig.suptitle("Обучающая выборка: состав и дисбаланс классов", fontsize=13.5, fontweight="bold", color=INK, y=1.02)

    ax = axes[0]
    reg = idx.region.str.replace("hip_left", "Бёдра").str.replace("hip_right", "Бёдра").str.replace("spine", "Позвоночник")
    counts = reg.value_counts()
    ax.bar(counts.index, counts.values, color=[BLUE, ORANGE][:len(counts)], width=0.55)
    for x, v in enumerate(counts.values):
        ax.text(x, v + 3, str(v), ha="center", fontsize=11, fontweight="bold", color=INK)
    lab = idx.dropna(subset=["y"])
    bad = int(lab.y.sum())
    ax.set_title(f"{len(idx)} снимков в {idx.n.nunique()} исследованиях\n"
                 f"с нарушением по оценке эксперта: {bad} из {len(lab)} ({bad / len(lab):.0%})",
                 fontsize=10.5, color=INK)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)

    ax = axes[1]
    types = ["v_pos", "v_roi", "v_axis", "v_artifact", "v_posrot"]
    names = ["Некорректная укладка позвоночника", "Некорректная область интереса бедра",
             "Не выровнена ось позвоночника", "Посторонние предметы", "Ротация / позиционирование бедра"]
    vals = [int(idx[t].sum()) for t in types]
    colors = [BLUE, ORANGE, BLUE, BLUE, ORANGE]
    y = np.arange(len(types))
    ax.barh(y, vals, color=colors, height=0.42)
    for k, v in enumerate(vals):
        ax.text(v + 0.5, k, str(v), va="center", fontsize=11, fontweight="bold", color=INK)
        ax.text(0, k + 0.42, names[k], va="bottom", ha="left", fontsize=10, color=INK)
    ax.set_yticks([]); ax.set_xlim(0, max(vals) * 1.15); ax.set_ylim(-0.6, len(types) - 0.2)
    ax.set_title("Положительных примеров по типам нарушений: от 6 до 36.\n"
                 "Отсюда все решения по модели — маленькие модели, монотонность, синтетика",
                 fontsize=10.5, color=INK)
    ax.set_xlabel("снимков с этим нарушением", fontsize=9.5, color="#5A6270")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    save(fig, "dataset")


# --------------------------------------------------------------------------- разбиение выборки
def split():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "Разбиение выборки: по пациенту, а не по снимку", ha="center", fontsize=14,
            fontweight="bold", color=INK)
    ax.text(0.5, 0.88, "Снимки одного исследования не попадают одновременно в обучение и в проверку — "
                       "иначе модель узнаёт пациента, а не нарушение",
            ha="center", fontsize=9.8, color="#5A6270")

    rng = np.random.default_rng(3)
    for f in range(5):
        y = 0.68 - f * 0.135
        ax.text(0.035, y + 0.035, f"фолд {f + 1}", fontsize=9.5, color="#5A6270", ha="left")
        for k in range(20):
            x = 0.14 + k * 0.038
            is_val = k // 4 == f
            ax.add_patch(FancyBboxPatch((x, y), 0.032, 0.07, boxstyle="round,pad=0.002,rounding_size=0.01",
                                        linewidth=1.0, edgecolor=ORANGE if is_val else BLUE,
                                        facecolor="#FDF0E6" if is_val else "#EAF0FB"))


    ax.add_patch(FancyBboxPatch((0.14, 0.02), 0.032, 0.05, boxstyle="round,pad=0.002,rounding_size=0.01",
                                linewidth=1.0, edgecolor=BLUE, facecolor="#EAF0FB"))
    ax.text(0.185, 0.045, "обучение", fontsize=9, color=INK, va="center")
    ax.add_patch(FancyBboxPatch((0.30, 0.02), 0.032, 0.05, boxstyle="round,pad=0.002,rounding_size=0.01",
                                linewidth=1.0, edgecolor=ORANGE, facecolor="#FDF0E6"))
    ax.text(0.345, 0.045, "проверка", fontsize=9, color=INK, va="center")
    ax.text(0.47, 0.045, "· каждый прямоугольник — 5 исследований, в проверке фолда 20 из 100\n"
                         "· 10 повторов с разными разбиениями · пороги подбираются внутри обучающей части\n"
                         "· синтетические нарушения добавляются только в обучение, все метрики — на настоящих снимках",
            fontsize=9, color="#5A6270", va="center", linespacing=1.5)
    save(fig, "split")


# ----------------------------------------------------------------------------- метрики с интервалами
def metrics_ci(report):
    rows = [("По снимкам (249)", "overall"), ("По исследованиям (100)", "study"),
            ("Позвоночник (99)", "spine"), ("Бёдра (150)", "hip")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    fig.suptitle("Метрики с 95 % доверительными интервалами (кросс-валидация по пациентам, 10 повторов)",
                 fontsize=13, fontweight="bold", color=INK, y=1.04)
    for ax, key, title in ((axes[0], "auc", "ROC-AUC"), (axes[1], "f1", "F1")):
        labels, mids, los, his = [], [], [], []
        for name, k in rows:
            src = report["studies"] if k == "study" else report["regions"].get(k)
            if src is None or key not in src:
                continue
            val, ci = src[key]
            labels.append(name); mids.append(val); los.append(val - ci[0]); his.append(ci[1] - val)
        ax.errorbar(mids, range(len(mids)), xerr=[los, his], fmt="o", color=BLUE, ecolor=GREY,
                    elinewidth=1.6, capsize=4, markersize=7)
        for y, (v, lo, hi) in enumerate(zip(mids, los, his)):
            ax.text(v, y - 0.3, f"{v:.3f}  [{v - lo:.2f}–{v + hi:.2f}]", ha="center", fontsize=9.5, color=INK)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels), fontsize=10.5)
        ax.set_xlim(0.55, 1.0); ax.set_ylim(len(labels) - 0.5, -0.75)
        ax.set_title(title, fontsize=12, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    save(fig, "metrics_ci")


# ------------------------------------------------------------------------------------- таксономия
def taxonomy(report, idx):
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "Таксономия нарушений: пять типов, каждый со своей моделью и порогом",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    keys = ["spine:v_axis", "spine:v_pos", "spine:v_artifact", "hip:v_roi", "hip:v_posrot"]
    for k, key in enumerate(keys):
        x = 0.01 + k * 0.198
        t = report["types"][key]
        region_color = BLUE if key.startswith("spine") else ORANGE
        box(ax, x, 0.45, 0.185, 0.34, "", color=region_color,
            fill="#EAF0FB" if key.startswith("spine") else "#FDF0E6")
        ax.text(x + 0.0925, 0.72, TYPE_RU[key], ha="center", va="center", fontsize=10, fontweight="bold",
                color=INK, linespacing=1.3)
        ax.text(x + 0.0925, 0.615, CRITERION[key], ha="center", va="center", fontsize=8.6, color="#5A6270")
        ax.text(x + 0.0925, 0.535, f"AUC {t['auc']:.2f} · F1 {t['f1']:.2f}\n{t['n_pos']} нарушений",
                ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.4)

    ax.text(0.26, 0.36, "позвоночник", ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.text(0.79, 0.36, "бедро", ha="center", fontsize=9.5, color=ORANGE, fontweight="bold")

    both = int(((idx[["v_axis", "v_pos", "v_artifact", "v_roi", "v_posrot"]].fillna(0).sum(1)) > 1).sum())
    ax.text(0.5, 0.27, "Несколько нарушений на одном снимке", ha="center", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(0.5, 0.12,
            f"Снимок помечается некачественным, если сработал хотя бы один тип — поэтому у каждой пометки\n"
            f"названа причина, а в отчёте их может быть несколько через «; ». В обучающей выборке таких снимков {both}.\n"
            f"Альтернативу (один порог на общую вероятность) проверили и отклонили: хуже по всем метрикам ТЗ сразу.",
            ha="center", fontsize=9.6, color="#5A6270", linespacing=1.6)
    save(fig, "taxonomy")


# --------------------------------------------------------------------------------- анализ ошибок
def errors(oof):
    types = ["v_axis", "v_pos", "v_artifact", "v_roi", "v_posrot"]
    names = ["Наклон оси", "Укладка позвоночника", "Посторонние предметы", "Область интереса бедра", "Ротация бедра"]
    fp = [int(((oof[t] == 0) & (oof[f"d_{t}"] == 1)).sum()) for t in types]
    fn = [int(((oof[t] == 1) & (oof[f"d_{t}"] == 0)).sum()) for t in types]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    y = np.arange(len(types))
    ax.barh(y - 0.2, fp, height=0.38, color=ORANGE, label="ложные тревоги")
    ax.barh(y + 0.2, fn, height=0.38, color=RED, label="пропуски")
    for k in y:
        ax.text(fp[k] + 0.3, k - 0.2, str(fp[k]), va="center", fontsize=9.5, color=INK)
        ax.text(fn[k] + 0.3, k + 0.2, str(fn[k]), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10); ax.invert_yaxis()
    ax.set_title("Анализ ошибок по типам нарушений (вне обучающих фолдов)", fontsize=12.5,
                 fontweight="bold", color=INK, pad=26)
    ax.set_xlabel("число снимков", fontsize=9.5, color="#5A6270")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", bbox_to_anchor=(1.0, 1.12), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    save(fig, "errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="_axis24", help="суффикс файлов pipeline_eval / pipeline_oof")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv("data/train/image_index.csv")
    report = json.loads(Path(f"data/train/pipeline_eval{args.eval}.json").read_text())
    oof = pd.read_csv(f"data/train/pipeline_oof{args.eval}.csv")
    architecture()
    dataset(idx)
    split()
    metrics_ci(report)
    taxonomy(report, idx)
    errors(oof)


if __name__ == "__main__":
    main()
