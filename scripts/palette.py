"""Одна палитра на все картинки: графики отчёта, слайды моделей и слайды со снимками.

Цвета лежат здесь, а не в каждом скрипте, чтобы перекраска под шаблон презентации была одной
правкой. Способов два:

1. положить рядом `docs/brand.json` с нужными ключами — он подхватится автоматически;
2. задать переменной среды, например `DXAQC_BRAND=docs/brand_dark.json`.

Файл может переопределять только часть ключей, остальные останутся по умолчанию:

    {"BLUE": "#1F4E9C", "ORANGE": "#D96B2B", "INK": "#101418"}

Значения — hex-строки. Смысл ключей:
    BLUE    основной акцент: позвоночник, шаги конвейера, крупные числа
    ORANGE  второй акцент: бедро, выделение
    GREEN   «хорошо»: выходы сервиса, корректная укладка
    RED     «плохо»: нарушения, пропуски, подсветка находки
    GREY    вспомогательные линии и стрелки
    INK     основной текст
    MUTED   пояснительный текст
    LINE    рамки и сетка
Плюс светлые заливки FILL_BLUE / FILL_ORANGE / FILL_GREEN — если их не задать, они считаются
автоматически как тот же цвет, осветлённый до фона.
"""
import json
import os
from pathlib import Path

DEFAULT = {
    "BLUE": "#4C7BD9",
    "ORANGE": "#E8833A",
    "GREEN": "#2E9E5B",
    "RED": "#C2413A",
    "GREY": "#9AA0A6",
    "INK": "#1B1D21",
    "MUTED": "#5A6270",
    "LINE": "#D8DCE2",
}


def _lighten(hex_color, amount=0.88):
    """Тот же оттенок, осветлённый до фона карточки."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(c + (255 - c) * amount)
    return "#{:02X}{:02X}{:02X}".format(mix(r), mix(g), mix(b))


def load():
    """Палитра с учётом docs/brand.json или файла из DXAQC_BRAND."""
    colors = dict(DEFAULT)
    path = Path(os.environ.get("DXAQC_BRAND", "docs/brand.json"))
    if path.exists():
        override = json.loads(path.read_text())
        unknown = set(override) - set(DEFAULT) - {"FILL_BLUE", "FILL_ORANGE", "FILL_GREEN"}
        if unknown:
            raise SystemExit(f"{path}: неизвестные ключи {sorted(unknown)}; допустимы {sorted(DEFAULT)}")
        colors.update(override)
    for name in ("BLUE", "ORANGE", "GREEN"):
        colors.setdefault(f"FILL_{name}", _lighten(colors[name]))
    return colors


def rgb(hex_color):
    """Hex → (R, G, B) для OpenCV-рисования поверх снимков."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


COLORS = load()
BLUE, ORANGE, GREEN, RED = COLORS["BLUE"], COLORS["ORANGE"], COLORS["GREEN"], COLORS["RED"]
GREY, INK, MUTED, LINE = COLORS["GREY"], COLORS["INK"], COLORS["MUTED"], COLORS["LINE"]
FILL_BLUE, FILL_ORANGE, FILL_GREEN = COLORS["FILL_BLUE"], COLORS["FILL_ORANGE"], COLORS["FILL_GREEN"]
