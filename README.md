# 以詞表擴充與兩階段微調為基之臺語歇後語情感極性分類

本專案以 [BERT-base-Chinese](https://huggingface.co/bert-base-chinese) 為骨幹，從 [T-BERT](https://huggingface.co/yixiuuu/tbert-base) 詞表中擷取臺語專有子詞並以其預訓練 Embedding 初始化，擴充模型對臺語詞的表徵能力。訓練採兩階段微調架構：Stage 1 凍結 Encoder 僅訓練 Embedding 與分類器；Stage 2 由頂層至底層漸進式解凍，搭配層級學習率衰減以保留預訓練知識。並進一步比較 R-Drop (Regularized Dropout) 一致性正則化與快速梯度法 (Fast Gradient Method, FGM) 對抗訓練對分類效能及跨種子穩定性的影響。

實驗以 2,268 筆臺語歇後語情感標註資料集（正面／中性／負面），於五組隨機種子 × 五折分層交叉驗證下比較六種模型配置。

- 分類類別：正面 (+1)、中性 (0)、負面 (-1)
- 模型配置：BERT-base-Chinese、T-BERT、Two-Stage、Two-Stage + R-Drop、Two-Stage + FGM、Two-Stage + R-Drop + FGM
- 實驗環境：NVIDIA GeForce RTX 2060
- 評估方式：五組隨機種子(0、1、42、123、1234) × 五折分層交叉驗證

---

## 專案結構

```
.
├── xiehouyu0.csv                # 歇後語資料集（2,268 筆，三類情感標籤）
│
├── cv_bert-base-chinese.py      # 基線：BERT-base-Chinese 單階段微調
├── cv_t-bert.py                 # 基線：T-BERT 單階段微調
├── cv_two_stage.py              # 提案：兩階段微調（詞表擴充 + 逐步解凍）
├── cv_two_stage_rdrop.py        # 提案 + R-Drop 一致性正則化
├── cv_two_stage_fgm.py          # 提案 + FGM 對抗式訓練
├── cv_two_stage_rdropfgm.py     # 提案 + R-Drop + FGM（最終模型）
│
├── analyze_token_lengths.py     # 分析 Tokenizer 編碼後的 Token 長度分佈
├── common_misclassified.py      # 找出六種模型共同預測錯誤的樣本
└── confusion_matrix.py          # 繪製五組隨機種子下的混淆矩陣
```

---

## 資料集

`xiehouyu0.csv` 包含 2,268 筆臺灣閩南語歇後語，欄位說明如下：

| 欄位         | 說明                                        |
| ------------ | ------------------------------------------- |
| `xie_hou_yu` | 歇後語文本                                  |
| `sentiment`  | 情感標籤（`1`＝正面、`2`＝中性、`3`＝負面） |

標籤分佈：正面 387 筆 (17.06%)、中性 577 筆 (25.44%)、負面 1,304 筆 (57.50%)。

### 資料來源

為確保標註品質，依正向、中性與負向類別比例進行分層隨機抽樣，抽取約 10% 樣本由第二位標註者獨立標記，以 Cohen's κ 評估一致性，意見相左者由第三人裁決。資料集整合以下四類來源，以「謎面──謎底」組合為唯一識別鍵去重複，得到 2,268 筆後進行人工標記：

1. **潘榮禮(2005)**《臺灣孽恝話新解》——取自[黃仲杰](https://hdl.handle.net/11296/pveak2) (2015) 依此書建立之情感標註資料集，共 1,089 筆
2. **林文平(2000)**《台灣歇後語典》——共 889 筆
3. **王永興(2015)**《台灣歇後語謔詰話》——因已絕版，依博客來公開書目頁面取得附有釋義之前 58 筆
4. **網路爬蟲**——以 Python Selenium 搭配 BeautifulSoup4 爬取下列公開網頁，共取得候選樣本 1,439 筆，經格式清理、去重複及人工審查後，保留 163 筆正向、265 筆中性樣本納入資料集

| 爬蟲網站名稱                                                                                                                                  | 資料筆數 | 擷取日期   |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------- |
| [臺北教師 e 教材－臺語歇後語（二）](https://tmrc.tiec.tp.edu.tw/HTML/RSR200811051155451DO/%E5%8F%B0%E8%AA%9E%E6%AD%87%E5%BE%8C%E8%AA%9E2.HTM) | 31       | 2025/11/20 |
| [臺北教師 e 教材－臺語歇後語（三）](https://tmrc.tiec.tp.edu.tw/HTML/RSR200811051155451DO/%E5%8F%B0%E8%AA%9E%E6%AD%87%E5%BE%8C%E8%AA%9E3.HTM) | 63       | 2025/11/20 |
| [臺北教師 e 教材－臺語歇後語（四）](https://tmrc.tiec.tp.edu.tw/HTML/RSR200811051155451DO/%E5%8F%B0%E8%AA%9E%E6%AD%87%E5%BE%8C%E8%AA%9E4.HTM) | 46       | 2025/11/20 |
| [嘉義縣教育資訊網－閩南歇後語](https://www.jhps.cyc.edu.tw/01/Site%20taiwan%20language/photo2.html)                                           | 133      | 2025/11/20 |
| [臺灣孽恝話（歇後語）](https://mypaper.pchome.com.tw/avun01/post/1339344122)                                                                  | 1,099    | 2025/11/20 |
| [Loxa 教育網閩南語歇後語](https://www.loxa.edu.tw/classweb/webView/index2.php?m_Id=14974&m_Type=3&m_Sort=1&webId=28549&teacher=yses-ml)       | 67       | 2025/11/20 |

---

## 預訓練模型下載

本專案使用兩個預訓練模型，程式執行時會透過 HuggingFace `transformers` 自動下載，也可以手動下載後指定本地路徑。

### bert-base-chinese

由 Google 發布的中文 BERT 模型，詞表規模達 21,128 個子詞單元。

```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-chinese'); AutoModel.from_pretrained('bert-base-chinese')"
```

HuggingFace 頁面：https://huggingface.co/bert-base-chinese

### T-BERT (tbert-base)

由[鍾明諺](https://hdl.handle.net/11296/aqxj79)開發，以臺灣國語、閩南語與客家語混合語料預訓練，詞表規模達 89,660 個子詞單元。訓練程式碼與語料來自 [DeepqEducation/t-bert](https://github.com/DeepqEducation/t-bert)；本研究直接載入發布於 HuggingFace 的預訓練模型權重。

```bash
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('yixiuuu/tbert-base'); AutoModel.from_pretrained('yixiuuu/tbert-base')"
```

HuggingFace 頁面：https://huggingface.co/yixiuuu/tbert-base  
原始 GitHub：https://github.com/DeepqEducation/t-bert  
T-BERT 論文：https://hdl.handle.net/11296/aqxj79

---

## 環境需求

### 實驗環境

本實驗於 NVIDIA GeForce RTX 2060 GPU 環境執行。

### 套件版本

| 套件         | 實驗使用版本 |
| ------------ | ------------ |
| Python       | 3.13.1       |
| PyTorch      | 2.9.0+cu126  |
| Transformers | 4.56.2       |
| CUDA         | 12.6         |
| scikit-learn | 1.9.0        |
| pandas       | 3.0.3        |
| numpy        | 2.5.1        |
| matplotlib   | 3.11.1       |
| seaborn      | 0.13.2       |
| openpyxl     | 3.1.5        |

> 若要確認目前環境版本，可執行：
>
> ```bash
> python --version
> python -c "import torch; print('torch:', torch.__version__)"
> python -c "import torch; print('cuda:', torch.version.cuda)"
> python -c "import transformers, sklearn, pandas, numpy, matplotlib, seaborn, openpyxl; libs=[transformers, sklearn, pandas, numpy, matplotlib, seaborn, openpyxl]; [print(lib.__name__, lib.__version__) for lib in libs]"
> ```

### 安裝指令

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn openpyxl
```

---

## 模型架構

### 基線模型

- **BERT-base-Chinese** (`bert-base-chinese`)：單階段微調基線。損失函數依各折訓練集類別分布計算類別權重，以加權交叉熵緩解正面／中性／負面樣本數不平衡問題，並搭配標籤平滑降低模型對單一類別的過度自信。
- **T-BERT** (`yixiuuu/tbert-base`)：以臺語語料預訓練的 BERT，同樣採單階段微調，損失函數設定與 BERT-base-Chinese 相同。

### 提案模型：兩階段微調 (Two-Stage Fine-tuning)

以 `bert-base-chinese` 為骨幹，從 T-BERT 詞表中篩選臺語專有子詞並擴充詞表，同時以 T-BERT 預訓練向量初始化新增 Embedding。

**Stage 1**：凍結 Encoder，僅訓練 Embedding、Pooler 與 Classifier。

**Stage 2**：分三個 Phase，由頂層往底層逐步解凍 Encoder，搭配逐層學習率衰減 (`decay^depth`)：

| Phase | 新增解凍層 | 累積解凍層範圍 | 頂層學習率 |
| ----- | ---------- | -------------- | ---------- |
| A     | 第 8–11 層 | 第 8–11 層     | 3.8e-5     |
| B     | 第 4–7 層  | 第 4–11 層     | 3.6e-5     |
| C     | 第 0–3 層  | 第 0–11 層     | 3.4e-5     |

**正則化與對抗訓練**：

| 模型                       | 額外技術                                  |
| -------------------------- | ----------------------------------------- |
| `cv_two_stage_rdrop.py`    | R-Drop（對稱 KL 散度約束，`alpha=0.5`）   |
| `cv_two_stage_fgm.py`      | FGM 對抗訓練（詞嵌入擾動，`epsilon=0.5`） |
| `cv_two_stage_rdropfgm.py` | 整合 R-Drop 與 FGM                        |

---

## 實驗設定

### 隨機種子

兩階段微調系列 (`cv_two_stage.py`、`cv_two_stage_rdrop.py`、`cv_two_stage_fgm.py`、`cv_two_stage_rdropfgm.py`) 預設跨五組隨機種子執行：

```python
seeds = (0, 1, 42, 123, 1234)
```

基線模型 (`cv_bert-base-chinese.py`、`cv_t-bert.py`) 預設 `seed=0`，搭配多組學習率搜尋：

```python
run_multi_lr_experiment(
    learning_rates=(2e-5, 3e-5, 4e-5, 5e-5),
    seed=0,
)
```

如需執行不同隨機種子 (0、1、42、123、1234)，請修改 `if __name__ == "__main__":` 區塊內的 `seed` 參數，每次執行一組。

### 詞頻篩選 (`min_new_token_freq`)

詞表擴充時，會先統計 T-BERT 專有子詞在訓練語料中的出現頻率，只有出現次數**大於或等於**此門檻的子詞才會被加入詞表。

設定位置：`CFG` 的 `min_new_token_freq`：

```python
@dataclass
class CFG:
    ...
    min_new_token_freq: int = 1
```

建議值參考：

- `1`：保留所有語料中出現的新詞（本實驗預設值）
- `2`：排除僅出現一次的極低頻子詞——論文實驗顯示，在兩階段微調模型 (`cv_two_stage.py`) 下此設定的改善案例多於退化案例，有助於提升分類效益

> 注意：此設定僅對兩階段微調系列有效。基線模型 (BERT-base-Chinese、T-BERT) 不涉及詞表擴充，無此參數。

---

## 實驗流程

所有訓練腳本採用相同的實驗框架：

1. 依隨機種子分層切出 **15% 保留測試集**（不參與任何 Epoch、超參數或詞頻門檻的選擇）
2. 對剩餘 85% 執行 **5-fold 分層交叉驗證**
3. 各折依**驗證集 Macro F1** 選擇最佳 checkpoint，取五折最佳 Epoch 的平均值作為重訓目標
4. 以完整訓練驗證集重新訓練最終模型，並在保留測試集上進行**最終評估**，回報 Accuracy、Macro F1、Weighted F1
5. 匯出各折指標、錯誤分類樣本(xlsx)與跨 Seed 彙總報告

---

## 使用方式

### 訓練

```bash
# 基線：BERT-base-Chinese
python cv_bert-base-chinese.py

# 基線：T-BERT
python cv_t-bert.py

# 提案：兩階段微調（跨五組 seed）
python cv_two_stage.py

# 提案 + R-Drop
python cv_two_stage_rdrop.py

# 提案 + FGM
python cv_two_stage_fgm.py

# 提案 + R-Drop + FGM
python cv_two_stage_rdropfgm.py
```

訓練過程的 stdout 會同時寫入對應的 log 檔（例如 `two-stage_rdropfgm_log.txt`）。訓練結果（模型 checkpoint、xlsx 報表）會分別儲存至各腳本對應的輸出資料夾（例如 `./two_stage_rdropfgm_runs/`、`./two_stage_rdropfgm_excel_outputs/`）。

### 分析工具

```bash
# 分析 Token 長度分佈，協助決定 max_length 設定
python analyze_token_lengths.py

# 繪製五組 seed 的混淆矩陣（輸出至 confusion_matrices/）
python confusion_matrix.py

# 找出六種模型共同預測錯誤的樣本（需先產生各模型的錯誤分類 xlsx）
python common_misclassified.py
```

---

## 主要超參數

| 參數                      | 值   |
| ------------------------- | ---- |
| `max_length`              | 40   |
| `train_batch_size`        | 16   |
| `n_folds`                 | 5    |
| `test_size`               | 0.15 |
| `label_smoothing`         | 0.05 |
| `warmup_ratio`            | 0.1  |
| `weight_decay`            | 0.01 |
| `early_stopping_patience` | 3    |
| R-Drop `alpha`            | 0.5  |
| FGM `epsilon`             | 0.5  |
| `phase_a_layerwise_decay` | 0.9  |
| `phase_b_layerwise_decay` | 0.9  |
| `phase_c_layerwise_decay` | 0.9  |

### `max_length` 設定依據

`max_length = 40` 的設定來自對資料集的 Token 長度統計分析。經 Tokenizer 編碼後（含 `[CLS]` 與 `[SEP]`），所有 2,268 筆歇後語的 Token 長度均介於 **10 至 35** 之間，`max_length = 40` 可完整涵蓋全部樣本而不發生截斷，長度不足的序列則以 padding 補齊至固定長度。
