"""Одна палитра на все картинки: графики отчёта, слайды моделей и слайды со снимками.

Цвета лежат здесь, а не в каждом скрипте, чтобы перекраска под шаблон презентации была одной
правкой. Тема выбирается переменной среды:

    DXAQC_THEME=brand   фирменная тёмная (по умолчанию): фон 15001D, акценты D66AFC и EBB5FE
    DXAQC_THEME=light   прежняя светлая — для README и печати

Отдельные цвета можно переопределить файлом `docs/brand.json` (или путём в `DXAQC_BRAND`):

    {"ACCENT": "#D66AFC", "ACCENT2": "#EBB5FE", "SURFACE": "#15001D"}

Почему именно эти цвета. Фирменная гамма одноцветная, и два ряда на графике рискуют слиться,
поэтому пара акцентов проверена валидатором на тёмном фоне: различие 17.5 ΔE при нормальном зрении
и 16.1 при дейтеранопии (порог 8), контраст к фону выше 3:1. Третий фиолетовый в ту же пару
не влезает (10.9 ΔE — уже неразличимо), поэтому статусные «хорошо» и «плохо» берутся из
согласованных по светлоте зелёного и розового и всегда сопровождаются подписью, а не только цветом.
"""
import json
import os
from pathlib import Path

BRAND = {                       # фирменная тёмная тема
    "SURFACE": "#15001D",       # фон слайда
    "PANEL": "#2E013E",         # заливка карточек и блоков
    "ACCENT": "#D66AFC",        # основной акцент: позвоночник, шаги конвейера, крупные числа
    "ACCENT2": "#EBB5FE",       # второй акцент: бедро, выделения
    "GOOD": "#57D9A3",          # «качественно», выходы сервиса
    "BAD": "#FF5C8A",           # нарушение, пропуск, подсветка находки
    "GREY": "#8E7F9B",          # стрелки и вспомогательные линии
    "INK": "#F4F0FA",           # основной текст
    "MUTED": "#B6A9C4",         # пояснительный текст
    "LINE": "#3A2247",          # рамки и сетка
}

LIGHT = {
    "SURFACE": "#FFFFFF",
    "PANEL": "#F5F6F7",
    "ACCENT": "#4C7BD9",
    "ACCENT2": "#E8833A",
    "GOOD": "#2E9E5B",
    "BAD": "#C2413A",
    "GREY": "#9AA0A6",
    "INK": "#1B1D21",
    "MUTED": "#5A6270",
    "LINE": "#D8DCE2",
}

THEMES = {"brand": BRAND, "light": LIGHT}


def _mix(hex_a, hex_b, t):
    """Смешать два цвета: t = 0 даёт первый, t = 1 второй."""
    a, b = hex_a.lstrip("#"), hex_b.lstrip("#")
    out = [round(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t) for i in (0, 2, 4)]
    return "#{:02X}{:02X}{:02X}".format(*out)


def load():
    name = os.environ.get("DXAQC_THEME", "brand")
    if name not in THEMES:
        raise SystemExit(f"неизвестная тема {name!r}; доступны {sorted(THEMES)}")
    colors = dict(THEMES[name])
    path = Path(os.environ.get("DXAQC_BRAND", "docs/brand.json"))
    if path.exists():
        override = json.loads(path.read_text())
        unknown = set(override) - set(BRAND)
        if unknown:
            raise SystemExit(f"{path}: неизвестные ключи {sorted(unknown)}; допустимы {sorted(BRAND)}")
        colors.update(override)
    # заливки карточек: подмешиваем акцент к фону, чтобы рамка и заливка были одного семейства
    for slot, key in (("FILL_ACCENT", "ACCENT"), ("FILL_ACCENT2", "ACCENT2"),
                      ("FILL_GOOD", "GOOD"), ("FILL_BAD", "BAD")):
        colors.setdefault(slot, _mix(colors["SURFACE"], colors[key], 0.12 if name == "brand" else 0.88))
    colors["THEME"] = name
    return colors


def rgb(hex_color):
    """Hex → (R, G, B) для OpenCV-рисования поверх снимков."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


COLORS = load()
THEME = COLORS["THEME"]
SURFACE, PANEL = COLORS["SURFACE"], COLORS["PANEL"]
INK, MUTED, LINE, GREY = COLORS["INK"], COLORS["MUTED"], COLORS["LINE"], COLORS["GREY"]
# прежние имена: ACCENT занимал место BLUE, ACCENT2 — ORANGE
BLUE, ORANGE, GREEN, RED = COLORS["ACCENT"], COLORS["ACCENT2"], COLORS["GOOD"], COLORS["BAD"]
FILL_BLUE, FILL_ORANGE = COLORS["FILL_ACCENT"], COLORS["FILL_ACCENT2"]
FILL_GREEN, FILL_BAD = COLORS["FILL_GOOD"], COLORS["FILL_BAD"]
