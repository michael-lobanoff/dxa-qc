# Manual review of spine proposals (blind to expert labels).
# FIX: point overrides (None = not in frame); UNSURE: faint structures; BOXES: foreign bodies.
FIX = {
 "001_spine": {"crest_a": [38, 285], "crest_b": [250, 273]},
 "003_spine": {"crest_a": [52, 257], "crest_b": [258, 262]},
 "004_spine": {"crest_b": [280, 258]},
 "005_spine": {"crest_a": [35, 282], "crest_b": [262, 275]},
 "007_spine": {"crest_a": [50, 287]},
 "008_spine": {"crest_a": [45, 275], "crest_b": [258, 278]},
 "009_spine": {"crest_a": [40, 268], "crest_b": [255, 262]},
 "010_spine": {"crest_a": [30, 283], "crest_b": [268, 272]},
 "011_spine": {"crest_a": [60, 292], "crest_b": [228, 294]},
 "014_spine": {"crest_a": [45, 300], "crest_b": [248, 300]},
 "015_spine": {"crest_a": [55, 278], "crest_b": [262, 284]},
 "016_spine": {"crest_a": [35, 257], "crest_b": [285, 247]},
}
UNSURE = {"003_spine", "005_spine", "008_spine", "010_spine", "011_spine", "014_spine"}
BOXES = {
 "002_spine": [[20, 2, 115, 28], [240, 0, 290, 20]],
 "005_spine": [[5, 2, 120, 35]],
 "010_spine": [[0, 5, 130, 50], [182, 0, 299, 45]],
 "011_spine": [[267, 267, 283, 283]],
 "014_spine": [[20, 2, 130, 30]],
 "022_spine": [[0, 5, 75, 30], [238, 85, 262, 112]],
 "031_spine": [[40, 58, 82, 82]],
 "037_spine": [[3, 10, 80, 32], [160, 3, 212, 35]],
 "047_spine": [[40, 60, 112, 112]],
 "048_spine": [[12, 45, 38, 72], [262, 45, 290, 72]],
 "054_spine": [[0, 0, 105, 35], [182, 0, 299, 35]],
 "055_spine": [[0, 0, 105, 30]],
 "059_spine": [[3, 0, 100, 50], [185, 0, 275, 40]],
 "063_spine": [[35, 78, 58, 102]],
 "066_spine": [[165, 0, 190, 12]],
 "079_spine": [[12, 0, 115, 25], [165, 0, 292, 40]],
 "085_spine": [[182, 0, 208, 48], [10, 2, 125, 28], [205, 5, 290, 28]],
 "087_spine": [[10, 0, 128, 55], [165, 0, 285, 50]],
 "091_spine": [[38, 32, 62, 58]],
 "100_spine": [[20, 0, 110, 110], [135, 0, 299, 100]],
}

FIX.update({
 "017_spine": {"crest_a": [30, 268], "crest_b": [262, 266]},
 "018_spine": {"crest_a": [30, 250], "crest_b": [270, 236]},
 "019_spine": {"crest_a": [45, 244], "crest_b": [255, 243]},
 "020_spine": {"crest_a": [40, 264], "crest_b": [268, 262]},
 "021_spine": {"crest_a": [45, 267], "crest_b": [245, 264]},
 "022_spine": {"crest_a": [50, 284], "crest_b": [262, 278]},
 "023_spine": {"crest_a": [30, 281], "crest_b": [262, 279]},
 "024_spine": {"crest_a": [40, 268], "crest_b": [262, 248]},
 "025_spine": {"crest_a": [52, 273], "crest_b": [262, 268]},
 "026_spine": {"crest_a": [45, 266], "crest_b": [250, 259]},
 "027_spine": {"crest_a": [38, 248], "crest_b": [268, 254]},
 "028_spine": {"crest_a": [70, 251], "crest_b": [262, 246]},
 "029_spine": {"crest_a": [15, 236], "crest_b": [262, 264]},
 "030_spine": {"crest_a": [35, 268], "crest_b": [262, 259]},
 "031_spine": {"crest_a": None, "crest_b": [268, 273]},
 "032_spine": {"crest_a": [22, 287], "crest_b": [268, 266]},
})
UNSURE |= {"017_spine", "020_spine", "021_spine", "025_spine", "030_spine", "031_spine", "032_spine"}

FIX.update({
 "033_spine": {"crest_a": [55, 253], "crest_b": [258, 250]},
 "035_spine": {"crest_a": [25, 272], "crest_b": [275, 266]},
 "036_spine": {"crest_a": [62, 270], "crest_b": [265, 267]},
 "037_spine": {"crest_a": [30, 273], "crest_b": [278, 263]},
 "038_spine": {"crest_a": [25, 293], "crest_b": [265, 285]},
 "039_spine": {"crest_a": [30, 259], "crest_b": [262, 265]},
 "041_spine": {"crest_a": [10, 254], "crest_b": [240, 257]},
 "042_spine": {"crest_a": [32, 266], "crest_b": [255, 277]},
 "044_spine": {"crest_a": [18, 263], "crest_b": [272, 258]},
 "046_spine": {"crest_a": [40, 259], "crest_b": [252, 252]},
 "047_spine": {"crest_a": [40, 268], "crest_b": [262, 266]},
 "048_spine": {"crest_a": [35, 254], "crest_b": [265, 250]},
})
UNSURE |= {"035_spine", "037_spine", "038_spine", "042_spine", "044_spine", "045_spine", "047_spine", "048_spine"}
ZOOM = {"040_spine"}

FIX.update({
 "049_spine": {"crest_a": [22, 266], "crest_b": [270, 263]},
 "050_spine": {"crest_a": [18, 262], "crest_b": [275, 241]},
 "051_spine": {"crest_a": [60, 257], "crest_b": [272, 248]},
 "052_spine": {"crest_a": [25, 292], "crest_b": [265, 290]},
 "053_spine": {"crest_a": [50, 273], "crest_b": [252, 272]},
 "055_spine": {"crest_a": [50, 269], "crest_b": [252, 268]},
 "056_spine": {"crest_a": [35, 290], "crest_b": [272, 268]},
 "057_spine": {"crest_a": [45, 300], "crest_b": None},
 "058_spine": {"crest_a": [40, 263], "crest_b": [262, 260]},
 "059_spine": {"crest_a": [25, 252], "crest_b": [262, 246]},
 "060_spine": {"crest_a": [35, 256], "crest_b": [255, 262]},
 "062_spine": {"crest_a": [50, 259], "crest_b": [255, 236]},
 "063_spine": {"crest_a": [55, 288], "crest_b": [250, 300]},
 "064_spine": {"crest_a": [40, 257], "crest_b": [250, 252]},
})
UNSURE |= {"052_spine", "053_spine", "056_spine", "057_spine", "058_spine", "060_spine", "063_spine"}
BOXES["049_spine"] = [[125, 160, 152, 202]]
ZOOM |= {"049_spine"}

FIX.update({
 "065_spine": {"crest_a": [62, 293], "crest_b": [250, 279]},
 "066_spine": {"crest_a": [55, 289], "crest_b": [255, 288]},
 "067_spine": {"crest_a": [60, 293], "crest_b": [240, 283]},
 "068_spine": {"crest_a": [40, 262], "crest_b": [265, 254]},
 "069_spine": {"crest_a": [40, 279], "crest_b": [262, 276]},
 "070_spine": {"crest_a": [25, 272], "crest_b": [272, 253]},
 "071_spine": {"crest_a": [55, 258], "crest_b": [248, 245]},
 "074_spine": {"crest_a": [38, 288], "crest_b": [268, 288]},
 "075_spine": {"crest_a": [40, 278], "crest_b": [262, 268]},
 "076_spine": {"crest_a": [65, 249], "crest_b": [255, 239]},
 "077_spine": {"crest_a": [62, 262], "crest_b": [275, 273]},
 "078_spine": {"crest_a": [25, 300], "crest_b": [265, 293]},
 "079_spine": {"crest_a": [20, 283], "crest_b": [262, 285]},
 "080_spine": {"crest_a": [40, 283], "crest_b": [262, 279]},
 "081_spine": {"crest_a": [35, 271], "crest_b": [240, 271]},
})
UNSURE |= {"066_spine", "068_spine", "075_spine", "079_spine"}

FIX.update({
 "082_spine": {"crest_a": [50, 272], "crest_b": [260, 268]},
 "083_spine": {"crest_a": [50, 238], "crest_b": [262, 233]},
 "084_spine": {"crest_a": [30, 262], "crest_b": [270, 257]},
 "085_spine": {"crest_a": [15, 258], "crest_b": [288, 247]},
 "086_spine": {"crest_a": [15, 249], "crest_b": [270, 253]},
 "087_spine": {"crest_a": [30, 281], "crest_b": [265, 278]},
 "088_spine": {"crest_a": [60, 265], "crest_b": [250, 266]},
 "089_spine": {"crest_a": [50, 238], "crest_b": [265, 241]},
 "090_spine": {"crest_a": [40, 229], "crest_b": [255, 231]},
 "091_spine": {"crest_a": [40, 293], "crest_b": [250, 283]},
 "092_spine": {"crest_a": [45, 233], "crest_b": [258, 240]},
 "093_spine": {"crest_a": [8, 280], "crest_b": [285, 257]},
 "094_spine": {"crest_a": [25, 263], "crest_b": [270, 270]},
 "095_spine": {"crest_a": [18, 265], "crest_b": [270, 251]},
 "096_spine": {"crest_a": [65, 232], "crest_b": [272, 240]},
 "097_spine": {"crest_a": [40, 223], "crest_b": [262, 226]},
 "098_spine": {"crest_a": [48, 233], "crest_b": [245, 236]},
 "099_spine": {"crest_a": [55, 298], "crest_b": [242, 283]},
 "100_spine": {"crest_a": [30, 298], "crest_b": [270, 298]},
})
UNSURE |= {"085_spine", "088_spine", "089_spine", "091_spine", "099_spine", "100_spine"}

FIX["040_spine"] = {"crest_a": [35, 230], "crest_b": [255, 245]}
UNSURE |= {"040_spine"}
BOXES["049_spine"] = [[112, 155, 145, 205]]
