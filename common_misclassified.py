# -*- coding: utf-8 -*-
"""
common_misclassified.py
=======================

找出六種情感極性分類模型在測試集中共同預測錯誤的樣本。

功能：
  - 讀取六個模型的測試集錯誤分類報告（xlsx）
  - 取六個模型錯誤樣本的交集
  - 將結果匯出為 CSV 檔案

輸入：
  各模型於指定隨機種子下的測試集錯誤分類報告，存放於 INPUT_DIR 資料夾。

輸出：
  common_errors_seed{SEED}.csv
"""

import os
from pathlib import Path

import pandas as pd


# =========================================================
# 設定
# =========================================================
SEED = 0
INPUT_DIR = Path(f"./test_common_errors/random{SEED}_mis_test")
OUTPUT_FILE = f"common_errors_seed{SEED}.csv"

# 六個模型的錯誤分類報告檔名
FILE_NAMES = {
    "BERT_Base":          f"bert_base_chinese_test_misclassified_seed{SEED}.xlsx",
    "T_BERT":             f"t_bert_test_misclassified_seed{SEED}.xlsx",
    "Two_Stage":          f"two_stage_test_misclassified_seed{SEED}.xlsx",
    "Two_Stage_RDrop":    f"two_stage_rdrop_test_misclassified_seed{SEED}.xlsx",
    "Two_Stage_FGM":      f"two_stage_fgm_test_misclassified_seed{SEED}.xlsx",
    "Two_Stage_RDrop_FGM": f"two_stage_rdropfgm_test_misclassified_seed{SEED}.xlsx",
}

# 對應輸出欄位名稱
PRED_COL_NAMES = {
    "BERT_Base":          "Pred_BERT_Base",
    "T_BERT":             "Pred_T-BERT",
    "Two_Stage":          "Pred_Two_Stage",
    "Two_Stage_RDrop":    "Pred_Two_Stage_RDrop",
    "Two_Stage_FGM":      "Pred_Two_Stage_FGM",
    "Two_Stage_RDrop_FGM": "Pred_Two_Stage_RDrop_FGM",
}


# =========================================================
# 資料載入
# =========================================================
dataframes = {}
for model_key, filename in FILE_NAMES.items():
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] 找不到檔案：{path}")
    dataframes[model_key] = pd.read_excel(path)
    print(f"  載入 {model_key}：{len(dataframes[model_key])} 筆錯誤樣本")


# =========================================================
# 取交集：找出六個模型皆預測錯誤的樣本
# =========================================================
sentence_sets = [set(df["Sentence"]) for df in dataframes.values()]
common_sentences = sentence_sets[0].intersection(*sentence_sets[1:])
print(f"\n共同錯誤樣本數：{len(common_sentences)}")


# =========================================================
# 整合各模型預測標籤
# =========================================================
# 以第一個模型（BERT_Base）的 True_Label 作為真實標籤來源
first_df = dataframes["BERT_Base"]
common_df = pd.DataFrame({"Sentence": list(common_sentences)})

common_df = common_df.merge(
    first_df[["Sentence", "True_Label", "Predicted_Label"]].rename(
        columns={"Predicted_Label": PRED_COL_NAMES["BERT_Base"]}
    ),
    on="Sentence",
    how="left",
)

# 依序合併其餘五個模型的預測標籤
for model_key in list(FILE_NAMES.keys())[1:]:
    df = dataframes[model_key]
    common_df = common_df.merge(
        df[["Sentence", "Predicted_Label"]].rename(
            columns={"Predicted_Label": PRED_COL_NAMES[model_key]}
        ),
        on="Sentence",
        how="left",
    )


# =========================================================
# 匯出結果
# =========================================================
common_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"結果已儲存至：{OUTPUT_FILE}")
