"""Ложные «посторонние предметы» на снимках с разметкой аппарата.

Публичный набор Mendeley (CC BY 4.0, другой денситометр): на снимках впечатаны рамки уровней и подписи
L1–L4, металла нет. Сейчас сервис на таких снимках оценку посторонних предметов выключает целиком,
потому что рамки и подписи похожи на тонкий металл. Здесь меряем, как часто тип сработал бы без этой
блокировки, — с прежней сетью сегментации и с обученной на разметке как отрицательном примере.

    python scripts/markup_false_alarms.py [папка моделей]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxaqc.service import QCService  # noqa: E402

warnings.filterwarnings("ignore")
SRC = Path("data/external/mendeley_dexa_spine")


def main(model_dir="models", n=60):
    svc = QCService(model_dir)
    files = sorted(SRC.glob("*.PNG"))[:n]
    printed = fired = analysed = 0
    for f in files:
        img = np.asarray(Image.open(f).convert("L"))
        r = svc.analyse(img, (0.6, 0.6))
        if r.get("unsupported") or r.get("blurred") or r.get("region") != "spine":
            continue
        analysed += 1
        if r["measurements"].get("printed_markup"):
            printed += 1
            # то, что вынес бы тип «посторонние предметы» без блокировки по разметке
            model = svc.decision["spine"]
            probs, decs, _ = model.predict(r["measurements"], svc.policy)
            fired += bool(decs["v_artifact"])
    print(f"{model_dir}: позвоночников {analysed}, с разметкой аппарата {printed}, "
          f"ложных «посторонних предметов» без блокировки: {fired} ({100 * fired / max(printed, 1):.0f} %)")


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
