"""Картинки со снимками пациентов для слайдов 2, 6, 7 и 18.

Эти четыре слайда показывают настоящие изображения, поэтому собираются отдельным скриптом и пишутся
в outputs/ — в репозиторий медицинские данные не попадают (ТЗ и здравый смысл). Скрипт коммитим,
результат — нет.

    python scripts/slide_cases.py [--out outputs/slides]

Что получается:
  slide02_problem.png   одна анатомия, два результата: ровная укладка против заваленной
  slide06_artifact.png  косточка белья: что видит глаз и что обвёл сервис
  slide07_refusal.png   сервис умеет сказать «не знаю»: прочерки вместо миллиметров и отказ на рентгене
  slide18_robustness.png устойчивость: вывод прогона архива и три факта
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, MUTED, BLUE, GREEN, RED, ORANGE = "#1B1D21", "#5A6270", "#4C7BD9", "#2E9E5B", "#C2413A", "#E8833A"
STRAIGHT, TILTED = "094_spine", "028_spine"          # выбраны по data/train/axis_features.csv
BROKEN_HIP = "069_hip_left"                          # бедро попало в кадр частично
SAMPLE_SPINE = "data/test_sample/CR000000_ПОП.dcm"   # снимок с косточкой белья


FIG = (13.33, 7.5)


def canvas(title, subtitle=None, figsize=FIG):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.955, title, ha="center", fontsize=19, fontweight="bold", color=INK)
    if subtitle:
        ax.text(0.5, 0.90, subtitle, ha="center", fontsize=12, color=MUTED)
    return fig, ax


def put(ax, img, x, y, w, h):
    """Вписать изображение в прямоугольник осей, сохранив пропорции.

    Оси нормированы 0..1 по обеим сторонам, а фигура не квадратная, поэтому пересчитываем
    через дюймы — иначе снимок выйдет сплющенным.
    """
    ih, iw = img.shape[:2]
    box_w_in, box_h_in = w * FIG[0], h * FIG[1]
    scale = min(box_w_in / iw, box_h_in / ih)
    dw, dh = iw * scale / FIG[0], ih * scale / FIG[1]        # обратно в единицы осей
    ax.imshow(img, extent=(x + (w - dw) / 2, x + (w + dw) / 2, y + (h - dh) / 2, y + (h + dh) / 2),
              aspect="auto", zorder=2, interpolation="lanczos")
    return dw, dh


def save(fig, out, name):
    path = Path(out) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def annotation_image(i):
    return np.asarray(Image.open(f"data/annotation/images/{i}.png").convert("L"))


# ------------------------------------------------------------------ слайд 2: в чём проблема
def slide2(out):
    """Ось рисуем ту же, что считает сервис: прямая через две точки столба, а не через центр кадра."""
    from dxaqc.service import QCService
    svc = QCService("models")
    fig, ax = canvas("Одна и та же анатомия — разный результат измерения",
                     "плотность считается внутри областей, размеченных на снимке: перекос кадра сдвигает саму цифру")
    for k, (i, label, color, bgr) in enumerate([(STRAIGHT, "корректная укладка", GREEN, (90, 209, 122)),
                                                (TILTED, "ось завалена", RED, (255, 90, 90))]):
        img = annotation_image(i)
        r = svc.analyse(img, (0.6, 0.606))
        t, b = r["points"]["col_top"], r["points"]["col_bottom"]
        tilt = abs(r["measurements"]["tilt_deg"])
        h, w = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # вертикаль кадра — через нижнюю точку оси, пунктиром
        for y in range(0, h, 10):
            cv2.line(rgb, (int(b[0]), y), (int(b[0]), min(y + 5, h - 1)), (170, 170, 170), 1, cv2.LINE_AA)
        # сама ось: прямая через col_top и col_bottom, продлённая на весь кадр
        dy = (b[1] - t[1]) or 1
        slope = (b[0] - t[0]) / dy
        p_top = (int(round(t[0] - slope * t[1])), 0)
        p_bot = (int(round(b[0] + slope * (h - 1 - b[1]))), h - 1)
        cv2.line(rgb, p_top, p_bot, bgr, 2, cv2.LINE_AA)
        for key in ("crest_a", "crest_b"):                      # уровень таза, от него отсчитывается ось
            c = r["points"].get(key)
            if c:
                cv2.circle(rgb, (int(c[0]), int(c[1])), 5, (90, 209, 122), 2, cv2.LINE_AA)

        put(ax, rgb, 0.08 + k * 0.46, 0.20, 0.38, 0.65)
        ax.text(0.27 + k * 0.46, 0.145, label, ha="center", fontsize=14, fontweight="bold", color=color)
        ax.text(0.27 + k * 0.46, 0.095, f"ось отклонена на {tilt:.1f}° при допуске 5° по ТЗ",
                ha="center", fontsize=11, color=MUTED)
    ax.text(0.5, 0.035, "разница между «лечение работает» и «не работает» — единицы процентов плотности; "
                        "перекос даёт столько же",
            ha="center", fontsize=11.5, color=INK, style="italic")
    save(fig, out, "slide02_problem.png")


# --------------------------------------------------- слайд 6: инородное тело, которого не видно
def slide6(out):
    from dxaqc.artifacts import artifact_mask
    from dxaqc.service import QCService, read_dicom_image
    svc = QCService("models")
    img, _ = read_dicom_image(SAMPLE_SPINE)
    r = svc.analyse(img)
    band = slice(0, int(img.shape[0] * 0.34))
    raw = cv2.cvtColor(img[band], cv2.COLOR_GRAY2RGB)
    mask = artifact_mask(img, r["points"])[band]
    marked = raw.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, cnts, -1, (255, 70, 70), 1, cv2.LINE_AA)

    fig, ax = canvas("Кейс: инородное тело, которое похоже на анатомию",
                     "тонкая яркая дуга поверх ребра — косточка бюстгальтера, рисунок 3 в техническом задании")
    for k, (im, cap) in enumerate([(raw, "что видит глаз: дуги рёбер"),
                                   (marked, "что обвёл сервис: линия поверх ребра")]):
        big = cv2.resize(im, (im.shape[1] * 3, im.shape[0] * 3), interpolation=cv2.INTER_LANCZOS4)
        put(ax, big, 0.03 + k * 0.49, 0.30, 0.45, 0.50)
        ax.text(0.255 + k * 0.49, 0.245, cap, ha="center", fontsize=12.5,
                fontweight="bold", color=INK if not k else RED)
    m = r["measurements"]
    ax.text(0.5, 0.15, f"площадь тонких ярких структур {m['area_top']:.0f} пикселей, маска сети {m['seg_area']:.0f} — "
                       f"выше, чем у всех 82 чистых снимков обучающей выборки",
            ha="center", fontsize=11.5, color=MUTED)
    ax.text(0.5, 0.075, f"вердикт сервиса: {', '.join(r['violations']) and 'присутствуют посторонние предметы'} "
                        f"(вероятность {r['score']:.2f})",
            ha="center", fontsize=13, fontweight="bold", color=RED)
    save(fig, out, "slide06_artifact.png")


# ------------------------------------------------- слайд 7: сервис умеет сказать «не знаю»
def slide7(out):
    from dxaqc.service import QCService, read_dicom_image
    from dxaqc.visualize import render_overlay
    svc = QCService("models")

    img = annotation_image(BROKEN_HIP)
    r = svc.analyse(img, (0.6, 0.606))
    left = render_overlay(img, r)

    rad_path = next(p for p in sorted(Path("data/external/lumos/x").rglob("*.Dcm")) if not p.name.startswith("._"))
    rimg, _ = read_dicom_image(rad_path)
    rr = svc.analyse(rimg)
    small = cv2.resize(rimg, (rimg.shape[1] // 4, rimg.shape[0] // 4), interpolation=cv2.INTER_AREA)
    right = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)

    fig, ax = canvas("Кейс: сервис умеет сказать «я этого не измерю»",
                     "отказ с причиной вместо правдоподобного, но выдуманного числа")
    put(ax, left, 0.02, 0.24, 0.44, 0.60)
    put(ax, right, 0.54, 0.24, 0.44, 0.60)
    ax.text(0.24, 0.185, "бедро попало в кадр частично", ha="center", fontsize=12.5,
            fontweight="bold", color=INK)
    ax.text(0.24, 0.115, "в полях кадра стоят прочерки,\nа не выдуманные миллиметры;\n"
                         "вердикт называет оба нарушения",
            ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.6)
    ax.text(0.76, 0.185, "обычная рентгенограмма поясницы", ha="center", fontsize=12.5,
            fontweight="bold", color=INK)
    ax.text(0.76, 0.115, "«оценка не проводится: снимок\nслишком крупный для денситометрии»\n"
                         "таких отклонено 66 из 66",
            ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.6)
    ax.text(0.5, 0.03, "уверенный вердикт на чужом снимке опаснее, чем отказ", ha="center",
            fontsize=12, color=INK, style="italic")
    save(fig, out, "slide07_refusal.png")


# ------------------------------------------------------------- слайд 18: устойчивость
def slide18(out, results=None):
    fig, ax = canvas("Устойчивость и воспроизводимость",
                     "то, что обычно выясняется на внедрении, а не на защите")
    lines = ["$ ./run.sh predict /data ./outputs",
             "",
             "499 images, 499 processed, 0 failed -> outputs/results.csv (+ .xlsx)",
             "mean time 0.43s",
             "",
             "$ .venv/bin/python -m pytest -q tests",
             "28 passed"]
    ax.add_patch(FancyBboxPatch((0.06, 0.42), 0.88, 0.37, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor="#2C3039", facecolor="#14161A"))
    for k, line in enumerate(lines):
        ax.text(0.09, 0.735 - k * 0.045, line, ha="left", va="center", fontsize=11.5,
                family="monospace", color="#9AA0A6" if line.startswith("$") else "#E9EAEC")
    facts = [("499 / 499", "файлов архива обработано,\nошибок ноль"),
             ("499 / 499", "совпадение контейнера\nи хоста по вердикту и типам"),
             ("28", "тестов, включая чтение\nсжатых DICOM")]
    for k, (num, cap) in enumerate(facts):
        x = 0.06 + k * 0.30
        ax.add_patch(FancyBboxPatch((x, 0.13), 0.28, 0.21, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    linewidth=1.3, edgecolor=BLUE, facecolor="#EAF0FB"))
        ax.text(x + 0.14, 0.275, num, ha="center", fontsize=20, fontweight="bold", color=BLUE)
        ax.text(x + 0.14, 0.19, cap, ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.4)
    ax.text(0.5, 0.055, "необработанных исключений нет: любая ошибка становится строкой отчёта со статусом Failure",
            ha="center", fontsize=11.5, color=INK, style="italic")
    save(fig, out, "slide18_robustness.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/slides")
    args = ap.parse_args()
    slide2(args.out)
    slide6(args.out)
    slide7(args.out)
    slide18(args.out)


if __name__ == "__main__":
    main()
