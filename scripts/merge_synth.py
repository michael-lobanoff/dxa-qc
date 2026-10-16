"""Сборка файла синтетики для решающего слоя из частей, сгенерированных по областям.

synth_decision.py с --regions пишет каждую область в свой файл; fit_decision.py и eval_pipeline.py
читают один. Отгружаемая модель обучена на ens3b = синтетика бёдер (поля кадра) + синтетика наклона
позвоночника (только виды tilt и tilt_ok: синтетику укладки позвоночника проверили и не берём).

Угол столба к тазу пуст, когда гребни вне кадра. Сервис в этом случае подставляет обычный наклон
(decision.spine_measurements) — здесь то же самое, иначе пропуск превратился бы в ноль, то есть
в «столб идеально параллелен тазу». Файлы, сгенерированные текущим кодом, пропусков уже не содержат.

    python scripts/merge_synth.py ens3_hip ens3_spine --out ens3b
"""
import argparse
from pathlib import Path

import pandas as pd

D = Path("data/train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="части: data/train/synth_meas_<part>.csv")
    ap.add_argument("--out", required=True, help="итог: data/train/synth_meas_<out>.csv")
    args = ap.parse_args()
    syn = pd.concat([pd.read_csv(D / f"synth_meas_{p}.csv") for p in args.parts], ignore_index=True)
    if "tilt_pelvis" in syn:
        spine = syn.abs_tilt.notna()
        syn.loc[spine, "tilt_pelvis"] = syn.loc[spine, "tilt_pelvis"].fillna(syn.loc[spine, "abs_tilt"])
    syn.to_csv(D / f"synth_meas_{args.out}.csv", index=False)
    print(syn.groupby(["kind", "label"]).size().to_string())


if __name__ == "__main__":
    main()
