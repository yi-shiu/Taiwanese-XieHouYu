# -*- coding: utf-8 -*-
"""
cv_t_bert.py
============

使用 T-BERT 進行情感極性分類之單階段微調實驗。

實驗流程：
1. 固定一組隨機種子。
2. 依序測試多組學習率。
3. 每組學習率執行 5-fold 分層交叉驗證。
4. 根據各折平均最佳 epoch，使用完整訓練驗證集重新訓練。
5. 在保留的 15% 測試集進行最終評估。
6. 匯出各折指標、錯誤分析、時間與學習率比較報告。

"""

import os
import sys
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

warnings.filterwarnings("ignore")


# =========================================================
# TeeLogger
# =========================================================
class TeeLogger:
    """同時將 stdout 輸出到終端機與 log 檔案。"""

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = TeeLogger("t_bert_log.txt")


# =========================================================
# 全域設定
# =========================================================
DEFAULT_SEED = 0
SEED = DEFAULT_SEED  # 每次執行由 set_global_seed() 覆寫


def set_global_seed(seed: int):
    global SEED
    SEED = seed
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 關閉 benchmark 模式並啟用確定性演算法，以提高實驗可重現性；
        # 若優先考量訓練速度，可將 benchmark 改為 True，但結果可能略有差異。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


set_global_seed(DEFAULT_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =========================================================
# 超參數設定
# =========================================================
@dataclass
class CFG:
    csv_path: str = "xiehouyu0.csv"
    text_col: str = "xie_hou_yu"
    label_col: str = "sentiment"

    model_name: str = "yixiuuu/tbert-base"

    num_labels: int = 3
    max_length: int = 40

    test_size: float = 0.15
    n_folds: int = 5

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    label_smoothing: float = 0.05

    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_train_epochs: int = 6

    early_stopping_patience: int = 3

    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.01

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # 模型與 checkpoint 輸出資料夾
    output_root: str = "./t_bert_runs"

    # Excel 報表統一輸出資料夾
    excel_output_dir: str = "./t_bert_excel_outputs"


cfg = CFG()
os.makedirs(cfg.output_root, exist_ok=True)
os.makedirs(cfg.excel_output_dir, exist_ok=True)


# =========================================================
# 資料集
# =========================================================
class XieHouYuDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =========================================================
# 自訂 Trainer
# =========================================================
class StrongTrainer(Trainer):
    """
    整合以下功能的自訂 HuggingFace Trainer：
      - 加權交叉熵損失（類別權重、標籤平滑）
    """

    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self._weights_moved = False
        self.loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """計算訓練損失。

        標準加權交叉熵損失；類別權重與標籤平滑透過 nn.CrossEntropyLoss 套用。
        """
        labels = inputs["labels"]
        clean_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        if not self._weights_moved and self.loss_fct.weight is not None:
            self.loss_fct.weight = self.loss_fct.weight.to(model.device)
            self._weights_moved = True

        outputs = model(**clean_inputs)
        logits = outputs.logits

        loss = self.loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# =========================================================
# 評估指標
# =========================================================
def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.argmax(preds, axis=1)

    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "macro_f1": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "weighted_f1": f1_weighted,
    }


# =========================================================
# 模型載入
# =========================================================
def load_model(cfg: CFG, source: Optional[str] = None) -> BertForSequenceClassification:
    """從指定來源或 cfg.model_name 載入 BertForSequenceClassification。"""
    model_source = source if source else cfg.model_name
    config = AutoConfig.from_pretrained(
        model_source,
        num_labels=cfg.num_labels,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
    )
    return BertForSequenceClassification.from_pretrained(model_source, config=config)


# =========================================================
# 記憶體清理
# =========================================================
def cleanup_memory():
    """釋放未使用的 GPU 快取與觸發 Python 垃圾回收。"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 報告與錯誤分析
# =========================================================
def print_data_split_info(total_count, trainval_count, test_count, labels_series):
    print("=" * 80)
    print("DATASET SPLIT INFORMATION")
    print("=" * 80)
    print(f"Total samples: {total_count}")

    print("Label distribution:")
    label_counts = labels_series.value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"  Label {label}: {count}")

    print("-" * 80)
    print(f"Train+Val samples (85%): {trainval_count}")
    print(f"Test samples (15%): {test_count}")
    print("=" * 80)


def append_misclassified_samples(trainer, dataset, raw_texts, fold_no, output_list):
    """收集驗證集分類錯誤樣本，供錯誤分析使用。"""
    pred_output = trainer.predict(dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            output_list.append({
                "Fold": fold_no,
                "Sentence": raw_texts[i],
                "True_Label": int(y_true[i]) + 1,
                "Predicted_Label": int(y_pred[i]) + 1,
            })


def print_cv_summary(fold_metrics, fold_best_epochs):
    """列印 CV 摘要並回傳各折平均最佳 epoch（四捨五入）。"""
    print("\n" + "=" * 110)
    print("K-FOLD CV SUMMARY (VALIDATION METRICS / BEST EPOCH)")
    print("=" * 110)
    print(
        f"{'Fold':<6} | "
        f"{'Accuracy':>10} | "
        f"{'Macro F1':>10} | "
        f"{'Precision(W)':>12} | "
        f"{'Recall(W)':>10} | "
        f"{'Weighted F1':>11} | "
        f"{'Best Epoch':>10}"
    )
    print("-" * 110)

    for i, (m, ep) in enumerate(zip(fold_metrics, fold_best_epochs), start=1):
        print(
            f"{i:<6} | "
            f"{m['accuracy']:>10.4f} | "
            f"{m['macro_f1']:>10.4f} | "
            f"{m['precision_weighted']:>12.4f} | "
            f"{m['recall_weighted']:>10.4f} | "
            f"{m['weighted_f1']:>11.4f} | "
            f"{ep:>10}"
        )

    print("-" * 110)
    avg_epoch = int(round(float(np.mean(fold_best_epochs))))
    print(
        f"{'Average':<6} | "
        f"{np.mean([m['accuracy'] for m in fold_metrics]):>10.4f} | "
        f"{np.mean([m['macro_f1'] for m in fold_metrics]):>10.4f} | "
        f"{np.mean([m['precision_weighted'] for m in fold_metrics]):>12.4f} | "
        f"{np.mean([m['recall_weighted'] for m in fold_metrics]):>10.4f} | "
        f"{np.mean([m['weighted_f1'] for m in fold_metrics]):>11.4f} | "
        f"{avg_epoch:>10}"
    )
    print("=" * 110)
    print(f"Retraining epochs: {avg_epoch}")

    return avg_epoch


def evaluate_final_test(
    model_path: str,
    tokenizer,
    test_ds: XieHouYuDataset,
    cfg: CFG,
    test_texts: Optional[List[str]] = None,
    seed: int = 0,
    learning_rate: Optional[float] = None,
):
    """載入最終模型，在測試集上評估並匯出錯誤分析報告。"""
    eval_start = time.time()
    model = load_model(cfg, source=model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_final_eval", report_to="none"),
        compute_metrics=compute_metrics,
    )
    pred_output = trainer.predict(test_ds)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids

    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET (15%)")
    print("=" * 80)
    print(classification_report(y_true + 1, y_pred + 1, digits=4, zero_division=0))
    print(
        f"FINAL EVALUATION ON TEST SET elapsed: {(time.time() - eval_start)/60:.2f} min")

    # 匯出測試集分類錯誤樣本
    if test_texts is not None:
        misclassified_test = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                misclassified_test.append({
                    "Sentence": test_texts[i],
                    "True_Label": int(y_true[i]) + 1,
                    "Predicted_Label": int(y_pred[i]) + 1,
                })
        mis_test_df = pd.DataFrame(misclassified_test)
        eval_lr = cfg.learning_rate if learning_rate is None else learning_rate
        eval_lr_tag = f"{eval_lr:.0e}".replace("-", "m")
        mis_test_xlsx = os.path.join(
            cfg.excel_output_dir,
            f"t_bert_lr{eval_lr_tag}_test_misclassified_seed{seed}.xlsx",
        )
        mis_test_df.to_excel(mis_test_xlsx, index=False)
        print(
            f"Test misclassification report saved: {mis_test_xlsx} ({len(mis_test_df)} samples)")

    test_metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
    }
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    test_metrics.update({
        "test_precision_macro": float(p_macro),
        "test_recall_macro": float(r_macro),
        "test_macro_f1": float(f1_macro),
        "test_precision_weighted": float(p_w),
        "test_recall_weighted": float(r_w),
        "test_weighted_f1": float(f1_w),
    })

    return trainer, test_metrics


# =========================================================
# 主程式：5-fold CV → 平均最佳 epoch → 完整訓練驗證集重新訓練 → 測試集評估
# =========================================================
def main(seed: int = 0):
    set_global_seed(seed)
    print("\n" + "#" * 100)
    print(
        f"# Experiment: seed = {seed}, "
        f"learning_rate = {cfg.learning_rate:.0e}"
    )
    print("#" * 100)

    total_start = time.time()
    lr_tag = f"{cfg.learning_rate:.0e}".replace("-", "m")

    # --- 資料載入 ---
    df = (
        pd.read_csv(cfg.csv_path)
        .dropna(subset=[cfg.text_col, cfg.label_col])
        .reset_index(drop=True)
    )
    texts = df[cfg.text_col].astype(str).tolist()
    # 將標籤從 1/2/3 轉換為 0/1/2 以符合 CrossEntropyLoss
    labels = (df[cfg.label_col].astype(int) - 1).tolist()
    labels_np = np.array(labels)

    # 依目前 seed 分層切出 15% 測試集（相同 seed 下切分保持一致）
    trainval_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=cfg.test_size,
        stratify=labels_np,
        random_state=SEED,
    )
    print_data_split_info(len(df), len(trainval_idx),
                          len(test_idx), df[cfg.label_col])

    # 匯出此 seed 的測試集樣本清單
    test_sample_rows = []
    for seq_no, raw_idx in enumerate(test_idx, start=1):
        test_sample_rows.append({
            "No": seq_no,
            "OriginalIndex": int(raw_idx),
            "sentiment": int(labels_np[raw_idx] + 1),
            "xie_hou_yu": texts[raw_idx],
        })
    test_sample_xlsx = os.path.join(
        cfg.excel_output_dir,
        f"t_bert_lr{lr_tag}_test_samples_seed{seed}.xlsx",
    )
    pd.DataFrame(test_sample_rows).to_excel(test_sample_xlsx, index=False)
    print(
        f"Test sample list saved: {test_sample_xlsx} ({len(test_sample_rows)} samples)")

    # --- Tokenizer 載入 ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # --- 測試集 Dataset ---
    test_texts_list = [texts[i] for i in test_idx]
    test_labels_list = [labels[i] for i in test_idx]
    test_ds = XieHouYuDataset(test_texts_list, test_labels_list,
                              tokenizer, cfg.max_length)

    # 剩餘 85% 用於分層 5-fold CV
    skf = StratifiedKFold(n_splits=cfg.n_folds,
                          shuffle=True, random_state=SEED)

    fold_metrics: List[Dict] = []
    fold_best_epochs: List[int] = []
    misclassified_val_data: List[Dict] = []
    fold_rows: List[Dict] = []

    cv_start = time.time()
    trainval_labels_np = labels_np[trainval_idx]

    for fold, (train_cv_idx, val_cv_idx) in enumerate(
        skf.split(trainval_idx, trainval_labels_np), start=1
    ):
        print(f"\n{'*' * 30} Fold {fold} / {cfg.n_folds} {'*' * 30}")

        train_idx_abs = trainval_idx[train_cv_idx]
        val_idx_abs = trainval_idx[val_cv_idx]

        train_texts = [texts[i] for i in train_idx_abs]
        val_texts = [texts[i] for i in val_idx_abs]
        train_labels = [labels[i] for i in train_idx_abs]
        val_labels = [labels[i] for i in val_idx_abs]

        # 標籤分佈（還原為原始 1/2/3 編碼）
        t_lbls, t_cnts = np.unique(
            np.array(train_labels) + 1, return_counts=True)
        v_lbls, v_cnts = np.unique(
            np.array(val_labels) + 1, return_counts=True)
        print(
            f"    Train: {dict(zip(t_lbls.tolist(), t_cnts.tolist()))} | "
            f"Val: {dict(zip(v_lbls.tolist(), v_cnts.tolist()))}"
        )

        train_ds = XieHouYuDataset(
            train_texts, train_labels, tokenizer, cfg.max_length)
        val_ds = XieHouYuDataset(
            val_texts, val_labels, tokenizer, cfg.max_length)

        # 使用折訓練集計算類別權重
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=np.array(train_labels),
        )
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float)

        print(
            f"    -> [Class Weights] Fold {fold} class weights: {np.round(class_weights, 4)}")

        fold_output_dir = os.path.join(
            cfg.output_root, f"lr_{lr_tag}_seed_{seed}_fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # 每折重新初始化模型
        model = load_model(cfg)

        args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            weight_decay=cfg.weight_decay,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            adam_epsilon=cfg.adam_epsilon,
            max_grad_norm=cfg.max_grad_norm,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=SEED,
        )

        trainer = StrongTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience)],
            class_weights=class_weights_tensor,
            label_smoothing=cfg.label_smoothing,
        )

        trainer.train()

        # 評估並記錄驗證集指標
        eval_result = trainer.evaluate(eval_dataset=val_ds)
        metrics = {
            "accuracy": eval_result["eval_accuracy"],
            "macro_f1": eval_result["eval_macro_f1"],
            "precision_macro": eval_result["eval_precision_macro"],
            "recall_macro": eval_result["eval_recall_macro"],
            "precision_weighted": eval_result["eval_precision_weighted"],
            "recall_weighted": eval_result["eval_recall_weighted"],
            "weighted_f1": eval_result["eval_weighted_f1"],
        }
        fold_metrics.append(metrics)

        print(
            f"[Fold {fold} Result] "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"F1-Macro={metrics['macro_f1']:.4f} | "
            f"F1-Weighted={metrics['weighted_f1']:.4f}"
        )

        # 從 log history 取得最佳 epoch
        best_epoch = cfg.num_train_epochs
        if trainer.state.best_model_checkpoint:
            try:
                best_step = int(
                    trainer.state.best_model_checkpoint.split("-")[-1])
                for log in trainer.state.log_history:
                    if log.get("step") == best_step and "epoch" in log:
                        best_epoch = log["epoch"]
                        break
            except Exception:
                pass
        # 備援：從 log history 找出 eval_macro_f1 最高的 epoch
        if best_epoch == cfg.num_train_epochs:
            best_f1 = -1.0
            for log in trainer.state.log_history:
                if "eval_macro_f1" in log and log["eval_macro_f1"] > best_f1:
                    best_f1 = log["eval_macro_f1"]
                    best_epoch = log.get("epoch", cfg.num_train_epochs)
        best_epoch = int(round(float(best_epoch)))
        fold_best_epochs.append(best_epoch)

        # 匯出各折指標列
        fold_rows.append({
            "Fold": fold,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision_macro": round(metrics["precision_macro"], 4),
            "Recall_macro": round(metrics["recall_macro"], 4),
            "Macro_F1": round(metrics["macro_f1"], 4),
            "Precision_weighted": round(metrics["precision_weighted"], 4),
            "Recall_weighted": round(metrics["recall_weighted"], 4),
            "Weighted_F1": round(metrics["weighted_f1"], 4),
            "Best_Epoch": best_epoch,
        })

        # 收集驗證集分類錯誤樣本，用於錯誤分析
        append_misclassified_samples(
            trainer,
            val_ds,
            val_texts,
            fold,
            misclassified_val_data,
        )

        del trainer, model
        cleanup_memory()

    cv_elapsed_min = (time.time() - cv_start) / 60
    print("\n" + "=" * 90)
    print(f"CV elapsed: {cv_elapsed_min:.2f} min")
    print("=" * 90)

    # 匯出各折指標
    cv_xlsx_name = os.path.join(
        cfg.excel_output_dir,
        f"t_bert_lr{lr_tag}_cv_each_fold_metrics_seed{seed}.xlsx",
    )
    cv_df = pd.DataFrame(fold_rows)
    cv_df.to_excel(cv_xlsx_name, index=False)
    print(f"CV fold metrics saved: {cv_xlsx_name}")

    # 匯出驗證集分類錯誤報告
    mis_val_df = pd.DataFrame(misclassified_val_data)
    mis_val_xlsx = os.path.join(
        cfg.excel_output_dir,
        f"t_bert_lr{lr_tag}_best_misclassified_val_seed{seed}.xlsx",
    )
    mis_val_df.to_excel(mis_val_xlsx, index=False)
    print(
        f"Validation misclassification report saved: {mis_val_xlsx} ({len(mis_val_df)} samples)")

    # 從各折 CV 平均最佳 epoch 推導重新訓練 epoch 目標
    avg_epoch = print_cv_summary(fold_metrics, fold_best_epochs)

    # 依據 CV 推導的 epoch 數，使用完整的訓練與驗證資料重新訓練模型
    print("\n" + "=" * 90)
    print(f"RETRAINING ON FULL TRAIN-VAL DATA (Epochs: {avg_epoch})")
    print("=" * 90)

    retrain_start = time.time()

    # 使用完整的訓練與驗證資料計算類別權重
    trainval_labels_list = [labels[i] for i in trainval_idx]
    retrain_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(trainval_labels_list),
        y=np.array(trainval_labels_list),
    )
    retrain_weights_tensor = torch.tensor(retrain_weights, dtype=torch.float)

    trainval_texts_list = [texts[i] for i in trainval_idx]
    trainval_ds = XieHouYuDataset(
        trainval_texts_list, trainval_labels_list, tokenizer, cfg.max_length)

    retrain_model = load_model(cfg)
    retrain_output_dir = os.path.join(
        cfg.output_root, f"lr_{lr_tag}_seed_{seed}_retrain_t_bert_model")

    retrain_args = TrainingArguments(
        output_dir=retrain_output_dir,
        num_train_epochs=avg_epoch,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    retrain_trainer = StrongTrainer(
        model=retrain_model,
        args=retrain_args,
        train_dataset=trainval_ds,
        tokenizer=tokenizer,
        class_weights=retrain_weights_tensor,
        label_smoothing=cfg.label_smoothing,
    )

    retrain_trainer.train()

    # 儲存重新訓練的最終模型
    final_model_path = os.path.join(retrain_output_dir, "best_model")
    retrain_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    retraining_elapsed_min = (time.time() - retrain_start) / 60
    print(f"RETRAINING elapsed: {retraining_elapsed_min:.2f} min")

    # 計算 5-fold 分層交叉驗證與完整訓練驗證集重新訓練的總時間；
    # 不包含最終測試集推論與評估時間。
    cv_retraining_time_min = cv_elapsed_min + retraining_elapsed_min
    print(
        f"CV + RETRAINING TIME:"
        f"{cv_retraining_time_min:.2f} min"
    )

    del retrain_trainer, retrain_model
    cleanup_memory()

    # 在保留的 15% 測試集上進行最終評估
    test_texts_raw = [texts[i] for i in test_idx]
    _, test_metrics = evaluate_final_test(
        final_model_path, tokenizer, test_ds, cfg,
        test_texts=test_texts_raw,
        seed=seed,
        learning_rate=cfg.learning_rate,
    )

    total_time_min = (time.time() - total_start) / 60
    print("\n" + "=" * 90)
    print(f"Total elapsed: {total_time_min:.2f} min")
    print("=" * 90)

    # 彙整此 seed 的結果
    val_summary = {
        "val_accuracy_mean":        float(np.mean([m["accuracy"] for m in fold_metrics])),
        "val_accuracy_std":         float(np.std([m["accuracy"] for m in fold_metrics])),
        "val_macro_f1_mean":        float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "val_macro_f1_std":         float(np.std([m["macro_f1"] for m in fold_metrics])),
        "val_weighted_f1_mean":     float(np.mean([m["weighted_f1"] for m in fold_metrics])),
        "val_weighted_f1_std":      float(np.std([m["weighted_f1"] for m in fold_metrics])),
    }

    seed_result = {
        "learning_rate": cfg.learning_rate,
        "seed": seed,
        **val_summary,
        **test_metrics,
        "cv_retraining_minutes": cv_retraining_time_min,
        "total_minutes": total_time_min,
    }
    return seed_result


# =========================================================
# 多學習率實驗（固定單一 Seed）
# =========================================================
def run_multi_lr_experiment(
    learning_rates=(2e-5, 3e-5, 4e-5, 5e-5),
    seed=0,
):
    """
    使用固定的一組 Seed，依序測試多組學習率。

    請自行調整：
      - learning_rates：例如 (2e-5, 3e-5, 4e-5, 5e-5)
      - seed：例如 0、1、42、123、1234
    """
    print("\n" + "*" * 110)
    print(
        f"* Multi-learning-rate experiment: "
        f"{len(learning_rates)} learning rates = {list(learning_rates)}, "
        f"fixed seed = {seed}"
    )
    print("*" * 110)

    experiment_start = time.time()
    lr_results = []

    for learning_rate in learning_rates:
        cfg.learning_rate = float(learning_rate)

        print("\n" + "#" * 110)
        print(
            f"# Running learning_rate={cfg.learning_rate:.0e}, "
            f"seed={seed}"
        )
        print("#" * 110)

        result = main(seed=seed)
        lr_results.append(result)

        # 每完成一組學習率就儲存部分結果，避免中斷遺失。
        _save_multi_lr_report(
            lr_results,
            seed=seed,
            suffix="partial",
        )

        cleanup_memory()

    total_min = (time.time() - experiment_start) / 60

    print("\n" + "*" * 110)
    print(
        f"* Multi-learning-rate experiment complete. "
        f"Total elapsed: {total_min:.2f} min"
    )
    print("*" * 110)

    _save_multi_lr_report(
        lr_results,
        seed=seed,
        suffix="final",
    )
    _print_multi_lr_summary(lr_results)

    return lr_results


def _save_multi_lr_report(lr_results, seed, suffix="final"):
    """儲存固定 Seed 下各學習率的完整比較結果。"""
    df = pd.DataFrame(lr_results)
    df = df.sort_values("learning_rate").reset_index(drop=True)
    df = df.round(6)

    out_name = os.path.join(
        cfg.excel_output_dir,
        f"t_bert_multi_lr_seed{seed}_report_{suffix}.xlsx",
    )
    df.to_excel(out_name, index=False)
    print(f"[Multi-learning-rate report] saved: {out_name}")


def _print_multi_lr_summary(lr_results):
    """列印固定 Seed 下各學習率的比較摘要。"""
    df = pd.DataFrame(lr_results).sort_values("learning_rate")

    print("\n" + "=" * 122)
    print("MULTI-LEARNING-RATE EXPERIMENT SUMMARY")
    print("=" * 122)
    print(
        f"{'Learning Rate':<16}"
        f"{'Seed':>8}"
        f"{'Val Acc':>10}"
        f"{'Val MacroF1':>14}"
        f"{'Val WgtF1':>12}"
        f"{'Test Acc':>11}"
        f"{'Test MacroF1':>15}"
        f"{'Test WgtF1':>13}"
        f"{'CV+Retrain(min)':>17}"
        f"{'Total(min)':>12}"
    )
    print("-" * 122)

    for _, row in df.iterrows():
        print(
            f"{row['learning_rate']:<16.0e}"
            f"{int(row['seed']):>8}"
            f"{row['val_accuracy_mean']:>10.4f}"
            f"{row['val_macro_f1_mean']:>14.4f}"
            f"{row['val_weighted_f1_mean']:>12.4f}"
            f"{row['test_accuracy']:>11.4f}"
            f"{row['test_macro_f1']:>15.4f}"
            f"{row['test_weighted_f1']:>13.4f}"
            f"{row['cv_retraining_minutes']:>17.2f}"
            f"{row['total_minutes']:>12.2f}"
        )

    print("=" * 122)


if __name__ == "__main__":
    # 固定一組 Seed，依序測試多組學習率。
    # 可自行調整 learning_rates 與 seed。
    run_multi_lr_experiment(
        learning_rates=(2e-5, 3e-5, 4e-5, 5e-5),
        seed=0,
    )
