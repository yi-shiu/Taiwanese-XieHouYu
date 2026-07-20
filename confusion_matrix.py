# -*- coding: utf-8 -*-
"""
confusion_matrix.py
===================

繪製六種情感極性分類模型於五組隨機種子下的混淆矩陣。

每組隨機種子會輸出一張圖片，每張圖片包含以下六種模型：

1. BERT-base-Chinese
2. T-BERT
3. Two-Stage
4. R-Drop Two-Stage
5. FGM Two-Stage
6. R-Drop + FGM Two-Stage

混淆矩陣的列代表實際標籤（Actual Label），欄代表預測標籤（Predicted Label）。

預設隨機種子：
- 0
- 1
- 42
- 123
- 1234

輸出目錄：
confusion_matrices/

輸出檔案：
- confusion_matrix_seed_0.png
- confusion_matrix_seed_1.png
- confusion_matrix_seed_42.png
- confusion_matrix_seed_123.png
- confusion_matrix_seed_1234.png
"""

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# =========================================================
# 全域設定
# =========================================================
OUTPUT_DIR = Path("confusion_matrices")
OUTPUT_DPI = 300

SEEDS: List[int] = [0, 1, 42, 123, 1234]
CLASS_LABELS: List[str] = ["Pos", "Neu", "Neg"]

MODEL_TITLES: List[str] = [
    "BERT-base-Chinese",
    "T-BERT",
    "Two-Stage (Ours)",
    "R-Drop Two-Stage (Ours)",
    "FGM Two-Stage (Ours)",
    "R-Drop + FGM Two-Stage (Ours)",
]

SUBPLOT_LABELS: List[str] = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


# =========================================================
# 五組 Seed 的混淆矩陣
#
# 每組 Seed 的模型順序皆為：
# 1. BERT-base-Chinese
# 2. T-BERT
# 3. Two-Stage
# 4. R-Drop Two-Stage
# 5. FGM Two-Stage
# 6. R-Drop + FGM Two-Stage
# =========================================================
CONFUSION_MATRICES_BY_SEED: Dict[int, List[np.ndarray]] = {
    0: [
        np.array([[44, 5, 9], [7, 61, 19], [4, 22, 170]]),
        np.array([[38, 7, 13], [3, 60, 24], [5, 19, 172]]),
        np.array([[45, 5, 8], [5, 61, 21], [6, 17, 173]]),
        np.array([[43, 7, 8], [5, 62, 20], [4, 14, 178]]),
        np.array([[44, 6, 8], [2, 64, 21], [4, 10, 182]]),
        np.array([[47, 5, 6], [4, 64, 19], [3, 12, 181]]),
    ],
    1: [
        np.array([[42, 1, 15], [3, 60, 24], [4, 13, 179]]),
        np.array([[36, 6, 16], [7, 55, 25], [9, 20, 167]]),
        np.array([[43, 6, 9], [4, 67, 16], [4, 15, 177]]),
        np.array([[41, 5, 12], [3, 64, 20], [2, 15, 179]]),
        np.array([[43, 5, 10], [6, 64, 17], [4, 12, 180]]),
        np.array([[42, 3, 13], [4, 63, 20], [3, 10, 183]]),
    ],
    42: [
        np.array([[48, 6, 4], [2, 67, 18], [7, 6, 183]]),
        np.array([[38, 3, 17], [1, 66, 20], [6, 9, 181]]),
        np.array([[46, 5, 7], [3, 66, 18], [1, 11, 184]]),
        np.array([[47, 8, 3], [5, 66, 16], [5, 10, 181]]),
        np.array([[48, 7, 3], [4, 71, 12], [5, 10, 181]]),
        np.array([[47, 10, 1], [2, 71, 14], [3, 9, 184]]),
    ],
    123: [
        np.array([[47, 3, 8], [5, 65, 17], [5, 14, 177]]),
        np.array([[45, 6, 7], [6, 60, 21], [5, 17, 174]]),
        np.array([[48, 6, 4], [7, 65, 15], [8, 8, 180]]),
        np.array([[48, 5, 5], [7, 62, 18], [6, 6, 184]]),
        np.array([[46, 5, 7], [5, 69, 13], [2, 8, 186]]),
        np.array([[46, 7, 5], [5, 68, 14], [4, 8, 184]]),
    ],
    1234: [
        np.array([[50, 3, 5], [6, 60, 21], [10, 15, 171]]),
        np.array([[47, 5, 6], [3, 61, 23], [7, 18, 171]]),
        np.array([[48, 5, 5], [1, 69, 17], [7, 20, 169]]),
        np.array([[47, 7, 4], [4, 67, 16], [8, 14, 174]]),
        np.array([[50, 5, 3], [5, 67, 15], [6, 14, 176]]),
        np.array([[50, 6, 2], [6, 68, 13], [6, 16, 174]]),
    ],
}


# =========================================================
# 圖表設定與資料驗證
# =========================================================
def configure_plot_style() -> None:
    """設定圖表的全域字體大小。"""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


def validate_inputs(
    matrices_by_seed: Mapping[int, Sequence[np.ndarray]],
    model_titles: Sequence[str],
    class_labels: Sequence[str],
) -> None:
    """檢查 Seed、模型數量及混淆矩陣維度是否符合設定。"""
    expected_shape = (len(class_labels), len(class_labels))

    if set(matrices_by_seed.keys()) != set(SEEDS):
        raise ValueError(
            "CONFUSION_MATRICES_BY_SEED 的 Seed 與 SEEDS 設定不一致。"
        )

    for seed, matrices in matrices_by_seed.items():
        if len(matrices) != len(model_titles):
            raise ValueError(
                f"Seed {seed} 的混淆矩陣數量為 {len(matrices)}，"
                f"但模型數量為 {len(model_titles)}。"
            )

        for model_index, matrix in enumerate(matrices):
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"Seed {seed}、模型 {model_titles[model_index]} 的矩陣維度"
                    f"為 {matrix.shape}，預期應為 {expected_shape}。"
                )


# =========================================================
# 混淆矩陣繪製
# =========================================================
def plot_seed_confusion_matrices(
    seed: int,
    matrices: Sequence[np.ndarray],
    output_dir: Path,
) -> Path:
    """
    繪製單一 Seed 的六模型混淆矩陣。

    圖片本身不顯示整體 Seed 標題，Seed 僅保留於輸出檔名中。
    """
    vmin = 0
    vmax = max(int(matrix.max()) for matrix in matrices)

    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.08],
        wspace=0.3,
        hspace=0.4,
    )

    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
    ]
    colorbar_axis = figure.add_subplot(grid[:, 3])

    heatmap_options = {
        "annot": True,
        "fmt": "d",
        "cmap": "Blues",
        "xticklabels": CLASS_LABELS,
        "yticklabels": CLASS_LABELS,
        "cbar": False,
        "vmin": vmin,
        "vmax": vmax,
    }

    for index, (axis, matrix, model_title) in enumerate(
        zip(axes, matrices, MODEL_TITLES)
    ):
        sns.heatmap(matrix, ax=axis, **heatmap_options)
        axis.set_title(
            f"{SUBPLOT_LABELS[index]} {model_title}",
            pad=10,
        )
        axis.set_xlabel("Predicted Label")

        # 每列只在最左側子圖顯示 Y 軸標籤，避免重複。
        if index in (0, 3):
            axis.set_ylabel("Actual Label")
        else:
            axis.set_ylabel("")
            plt.setp(axis.get_yticklabels(), visible=False)

    # 六張子圖共用同一個 Colorbar。
    figure.colorbar(
        axes[-1].collections[0],
        cax=colorbar_axis,
    )

    output_path = output_dir / f"confusion_matrix_seed_{seed}.png"
    figure.savefig(
        output_path,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


# =========================================================
# 主程式
# =========================================================
def main() -> None:
    """依序輸出五組 Seed 的混淆矩陣圖片。"""
    validate_inputs(
        CONFUSION_MATRICES_BY_SEED,
        MODEL_TITLES,
        CLASS_LABELS,
    )
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        output_path = plot_seed_confusion_matrices(
            seed=seed,
            matrices=CONFUSION_MATRICES_BY_SEED[seed],
            output_dir=OUTPUT_DIR,
        )
        print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
