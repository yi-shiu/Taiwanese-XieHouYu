# -*- coding: utf-8 -*-
"""
analyze_token_lengths.py
========================
分析歇後語資料集經 Tokenizer 編碼後的 Token 長度，
並協助決定模型訓練時的 max_length 設定。

功能：
  - 統計不含與包含特殊詞元的 Token 長度
  - 顯示最短、最長及平均 Token 長度
  - 分析不同 max_length 候選值下的截斷樣本數與比例
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from transformers import BertTokenizer


# =========================================================
# 預設設定
# =========================================================
DEFAULT_DATA_PATH = "./xiehouyu0.csv"
DEFAULT_TEXT_COLUMN = "xie_hou_yu"
DEFAULT_TOKENIZER_NAME = "bert-base-chinese"  # yixiuuu/tbert-base
DEFAULT_MAX_LEN_CANDIDATES = [16, 24, 32, 40, 48]

# =========================================================
# 長度分析
# =========================================================


def analyze_token_lengths(
    texts: List[str],
    tokenizer: BertTokenizer,
) -> Tuple[np.ndarray, np.ndarray]:
    """計算每筆文本的 token 數量。

    Args:
        texts:     原始文本列表。
        tokenizer: 已載入的 BertTokenizer 實例。

    Returns:
        token_lengths:   不含特殊詞元的子詞數量陣列。
        encoded_lengths: 包含 [CLS]、[SEP] 等特殊詞元的實際輸入長度陣列。
    """
    token_lengths: List[int] = []
    encoded_lengths: List[int] = []

    for text in texts:
        text = str(text)

        # 純子詞切分長度（不含 special tokens）
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))

        # 實際送入模型的序列長度（含 [CLS] 與 [SEP]）
        encoded = tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        encoded_lengths.append(len(encoded["input_ids"]))

    return np.array(token_lengths), np.array(encoded_lengths)


# =========================================================
# 報表輸出
# =========================================================
def print_length_statistics(lengths: np.ndarray, name: str = "Encoded Length") -> None:
    """列印長度統計摘要（最短、最長、平均）。

    Args:
        lengths: 長度數值陣列。
        name:    顯示在標題中的欄位名稱。
    """
    print(f"\n{'=' * 80}")
    print(f"{name} statistics")
    print(f"{'=' * 80}")
    print(f"  最短長度 : {lengths.min()}")
    print(f"  最長長度 : {lengths.max()}")
    print(f"  平均長度 : {lengths.mean():.2f}")


def print_truncation_analysis(
    lengths: np.ndarray,
    max_len_candidates: List[int],
) -> None:
    """列印各 max_length 候選值下的截斷比例。

    Args:
        lengths:            實際編碼長度陣列（含 special tokens）。
        max_len_candidates: 待評估的 max_length 候選值列表。
    """
    print(f"\n{'=' * 80}")
    print("Truncation analysis")
    print(f"{'=' * 80}")
    for max_len in max_len_candidates:
        n_truncated = int(np.sum(lengths > max_len))
        ratio = n_truncated / len(lengths) * 100
        print(
            f"  max_length = {max_len:>3}  →  截斷樣本數: {n_truncated:>5} / {len(lengths)}"
            f"  ({ratio:5.2f}%)"
        )


# =========================================================
# CLI 介面
# =========================================================
def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="分析 BERT tokenizer 在歇後語資料集上的 token 長度分佈。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA_PATH,
        help="CSV 資料檔路徑。",
    )
    parser.add_argument(
        "--text-col", type=str, default=DEFAULT_TEXT_COLUMN,
        help="文本欄位名稱。",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME,
        help="HuggingFace tokenizer 名稱或本地路徑。",
    )
    parser.add_argument(
        "--max-len-candidates", type=int, nargs="+",
        default=DEFAULT_MAX_LEN_CANDIDATES,
        metavar="N",
        help="待評估的 max_length 候選值（可指定多個）。",
    )
    return parser.parse_args()


# =========================================================
# 主程式
# =========================================================
def main() -> None:
    args = parse_args()

    # --- 資料載入 ---
    if not os.path.exists(args.data):
        print(f"[ERROR] 找不到資料檔案：{args.data}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.data)

    if args.text_col not in df.columns:
        print(
            f"[ERROR] 欄位 '{args.text_col}' 不存在。"
            f"現有欄位：{list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    texts: List[str] = df[args.text_col].fillna("").astype(str).tolist()

    print("=" * 80)
    print("Dataset loaded")
    print("=" * 80)
    print(f"  資料筆數  : {len(texts)}")
    print(f"  Tokenizer : {args.tokenizer}")

    # --- Tokenizer 載入 ---
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    # --- 長度分析 ---
    token_lengths, encoded_lengths = analyze_token_lengths(texts, tokenizer)

    print_length_statistics(
        token_lengths,   name="Token length（不含 special tokens）")
    print_length_statistics(
        encoded_lengths, name="Encoded length（含 special tokens）")

    print_truncation_analysis(encoded_lengths, args.max_len_candidates)


if __name__ == "__main__":
    main()
