# Поиск внешних данных: что нашли и что запрашиваем

Искали снимки, максимально близкие к нашим: денситометрия (не обычный рентген), поясничный отдел
в прямой проекции и проксимальный отдел бедра. Прочесаны Zenodo, Figshare, Harvard Dataverse, Dryad,
Mendeley Data, Kaggle, GitHub, Roboflow и статьи через Europe PMC (по разделам «доступность данных»).

## Итог поиска

| Набор | Что внутри | Насколько близко | Доступ |
|---|---|---|---|
| **DEXSIT** (Sethu Institute of Technology, Индия) | DXA: AP позвоночник + **левое и правое бедро**, 42 пациента × 3 снимка (126) в статье 2018, на сайте заявлено 441; PNG; аннотации T/Z-score, BMD, площадь, возраст, пол | **Точное попадание** — единственный открытый источник с DXA **бёдер** | Подписать лицензию и написать с рабочей почты |
| **Bone Densitometry Dataset** (Арак, Иран) | DXA шейки бедра и позвоночника, 3643 пациента | Очень близко, но нет публичной ссылки | Письмо авторам препринта |
| Osteoporosis DEXA Spine (Мултан, Пакистан) | 351 DXA позвоночника, PNG, CC BY 4.0 | Близко, но только позвоночник | **Уже скачано**, лежит в `data/external/` |
| BUU-LSPINE (Таиланд) | 3600 пациентов, поясничный отдел, обычный рентген, разметка краёв L1–L5 | Другая модальность | Заявка через форму |
| VinDr-SpineXR | 10 469 снимков позвоночника, обычный рентген | Другая модальность | PhysioNet, аккредитация |
| NHANES, UK Biobank | Только числа плотности, изображения по отдельной заявке на проект | — | Долго |

Чего в открытом доступе нет ни у кого: **разметки качества укладки**. Поэтому ни один из этих наборов
не поднимет F1 напрямую — их ценность в проверке устойчивости на чужом аппарате и в том, что об этом
можно честно рассказать на питче.

## Письмо 1. Запрос DEXSIT

Отправлять с рабочей почты организации (студенческие адреса они не принимают), приложив подписанную
лицензию: <http://sethu.ac.in/wp-content/uploads/2018/12/LICENSE-AGREEMENT.pdf>.
Лицензия разрешает **некоммерческое научное использование**, запрещает передачу третьим лицам
и требует ссылки на статью ICBSII 2018.

**Кому:** tamilselvi@sethu.ac.in
**Копия:** naziafathima@sethu.ac.in, parisabeham@sethu.ac.in
**Тема:** Request for access to the DEXSIT database (signed EULA attached)

> Dear Dr. Tamilselvi,
>
> I am writing to request access to the DEXSIT database described in your ICBSII 2018 paper
> "DEXSIT: A Benchmark Database for BMD Measurement and Analysis".
>
> Our team is developing an automated quality-control tool for DXA studies: given an AP lumbar spine
> or proximal femur scan, it checks patient positioning against the standard criteria (spine axis
> alignment, coverage from mid-Th12 to the iliac crests, femoral rotation judged by the lesser
> trochanter, field margins around the region of interest) and reports what is wrong and why.
>
> Our current data come from a single scanner model, so we are looking for images from a different
> centre to check that the tool degrades gracefully rather than silently. DEXSIT is the only public
> collection we have found that contains DXA images of the AP spine together with both femurs, which
> is exactly the combination we need. We would use the images only to validate robustness — not for
> any commercial purpose, and we would not redistribute them.
>
> The signed End User License Agreement is attached. I would be glad to acknowledge DEXSIT and cite
> the ICBSII 2018 paper in any resulting report or publication.
>
> Thank you for making the database available.
>
> Best regards,
> <имя, должность, организация, рабочая почта>

## Письмо 2. Запрос иранского набора

Препринт: <https://www.medrxiv.org/content/10.1101/2025.01.25.24319689v1> (Masnabadi, Sadeghi-Niaraki,
Karimi, AbuHmed, Azarbani, Choi). Публичной ссылки на данные в препринте нет — пишем авторам.

**Тема:** Access to the bone densitometry image dataset (medRxiv 2025.01.25.24319689)

> Dear Authors,
>
> I read your preprint "Bone Densitometry Dataset for Computer Aided Osteoporosis Disease Detection"
> describing the collection from the Arak bone densitometry centre. I could not find a public link to
> the images and would like to ask whether the dataset, or a subset of it, can be shared for research.
>
> We are building an automated quality-control tool for DXA studies — it checks patient positioning
> (spine axis, coverage, femoral rotation, field margins) and explains the finding to the technologist.
> All our training data come from one scanner model, so scans from another centre would let us measure
> how well the tool transfers. We are interested in the femoral neck and spine scan images; the BMD
> values themselves are not needed.
>
> We would be happy to sign a data use agreement, to use the data solely for non-commercial research,
> not to redistribute it, and to cite your work.
>
> Thank you for considering this.
>
> Best regards,
> <имя, должность, организация, рабочая почта>

## Что делать с данными, когда придут

1. Прогнать `scripts/check_external.py` — доля принятых сервисом, определение области, распределение
   новизны. Для DEXSIT это **первая внешняя проверка бедра**: до сих пор внешне мы проверяли только
   позвоночник.
2. Посмотреть глазами наложения на 10–15 снимках: где встают точки, не съезжает ли выравнивание кропа.
3. Если аппарат другой и что-то ломается — чинить так же, как починили впечатанную разметку: находить
   причину, измерять эффект на своих данных, фиксировать в отчёте.
4. В презентацию — строка «проверено на внешних данных N центров, включая бёдра».
