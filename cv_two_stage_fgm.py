# -*- coding: utf-8 -*-
"""
cv_two_stage_fgm.py
===================

結合 FGM 對抗式訓練的兩階段微調實驗。

模型與詞表設定：
  以 bert-base-chinese 為基礎模型，使用 T-BERT 專有子詞進行詞表擴充，
  並以 T-BERT 對應的預訓練向量初始化新增子詞的 Embedding。

兩階段訓練策略：
  Stage 1 — 凍結 Encoder，僅訓練 Embedding、Pooler 與 Classifier。
  Stage 2 — 由頂層往底層逐步解凍 Encoder
            （Phase A → Phase B → Phase C，每次新增解凍 4 層），
            搭配逐層學習率衰減（decay^depth）。

對抗式訓練方法：
  - FGM：在詞嵌入上施加對抗擾動，取乾淨與對抗梯度平均值更新參數。

實驗流程：
  1. 依序執行多組隨機種子實驗（預設 5 組）。
  2. 每組種子執行 5-fold 分層交叉驗證。
  3. 根據各折最佳 epoch 的平均值，使用完整訓練驗證集重新訓練模型。
  4. 在預先保留的 15% 測試集上進行最終評估。
  5. 匯出各折指標、錯誤分類樣本、測試結果與跨 Seed 彙總報告。
"""

import sys
import os
import shutil
import random
import time
import warnings
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

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


sys.stdout = TeeLogger("two-stage_fgm_log.txt")

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

    base_model_name: str = "bert-base-chinese"
    extra_tokenizer_name: Optional[str] = "yixiuuu/tbert-base"
    init_checkpoint: Optional[str] = None

    num_labels: int = 3

    max_length: int = 40

    test_size: float = 0.15
    n_folds: int = 5
    min_new_token_freq: int = 1  # 在此設定詞頻門檻：出現次數低於此值的 T-BERT 新詞將被排除

    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1

    label_smoothing: float = 0.05

    train_batch_size: int = 16
    eval_batch_size: int = 32

    use_fgm: bool = True
    fgm_epsilon: float = 0.5
    fgm_enabled_phases: Tuple[str, ...] = (
        "Stage 1", "Phase A", "Phase B", "Phase C")

    # Stage 1：凍結 Encoder，僅訓練 Embedding、Pooler 與 Classifier
    # Stage 2：由頂層往底層逐步解凍 Encoder，每個 Phase 新解凍 4 層
    epochs_stage_1: int = 4
    epochs_phase_a: int = 4
    epochs_phase_b: int = 4
    epochs_phase_c: int = 10

    early_stopping_patience: int = 3

    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    weight_decay: float = 0.01

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Stage 1 學習率：透過梯度縮放近似降低既有詞更新幅度，新詞保留完整梯度
    stage_1_head_lr: float = 4e-5
    stage_1_emb_lr: float = 3.6e-5       # 既有詞
    stage_1_new_token_lr: float = 4e-5   # T-BERT 初始化的新詞

    # Stage 2（Phase A–C）：學習率由頂層往底層逐層幾何衰減（0.9^depth）；
    # Embedding 學習率隨累積解凍層數遞減；分類頭學習率固定。
    phase_a_head_lr: float = 3.8e-5
    phase_a_top_layer_lr: float = 3.8e-5
    phase_a_layerwise_decay: float = 0.9
    phase_a_unfreeze_layers: Tuple[int, ...] = (8, 9, 10, 11)

    phase_b_head_lr: float = 3.6e-5
    phase_b_top_layer_lr: float = 3.6e-5
    phase_b_layerwise_decay: float = 0.9
    phase_b_unfreeze_layers: Tuple[int, ...] = (4, 5, 6, 7)

    phase_c_head_lr: float = 3.4e-5
    phase_c_top_layer_lr: float = 3.4e-5
    phase_c_layerwise_decay: float = 0.9
    phase_c_unfreeze_layers: Tuple[int, ...] = (0, 1, 2, 3)

    # 模型與 checkpoint 輸出資料夾
    output_root: str = "./two_stage_fgm_runs"

    # Excel 報表統一輸出資料夾
    excel_output_dir: str = "./two_stage_fgm_excel_outputs"

    keep_only_final_phase_per_fold: bool = True


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
# FGM 對抗擾動
# =========================================================
class FGM:
    """
    在詞嵌入上進行 FGM 對抗式訓練（Miyato et al., 2017）。

    每個訓練步驟：
    1. 計算乾淨損失並反向傳播，累積乾淨梯度。
    2. 備份乾淨梯度；執行獨立的前向與反向傳遞，以取得攻擊方向。
    3. 透過 attack() 擾動 Embedding；還原乾淨梯度；計算對抗損失並反向傳播。
    4. 呼叫 restore() 還原 Embedding；優化器使用乾淨梯度與對抗梯度的平均值更新參數。
    """

    def __init__(self, model: nn.Module, emb_name: str = "word_embeddings"):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}

    def attack(self, epsilon: float = 0.5):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if torch.isfinite(norm) and norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =========================================================
# 自訂 Trainer
# =========================================================
class StrongTrainer(Trainer):
    """
    整合以下功能的自訂 HuggingFace Trainer：
      - 加權交叉熵損失（類別權重、標籤平滑）
      - 逐 Phase 分層學習率優化器組（透過 optimizer_grouped_parameters 傳入）
      - FGM 對抗訓練（覆寫 training_step）
    """

    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        optimizer_grouped_parameters: Optional[List[Dict]] = None,
        use_fgm: bool = False,
        fgm_epsilon: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.optimizer_grouped_parameters = optimizer_grouped_parameters

        self.use_fgm = use_fgm
        self.fgm_epsilon = fgm_epsilon
        self.fgm = FGM(self.model) if use_fgm else None
        self._weights_moved = False
        self.loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    def create_optimizer(self):
        if self.optimizer is None:
            if self.optimizer_grouped_parameters is None:
                return super().create_optimizer()
            self.optimizer = torch.optim.AdamW(
                self.optimizer_grouped_parameters,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """計算訓練損失。

        使用加權交叉熵計算損失，並套用類別權重與標籤平滑。
        """
        labels = inputs["labels"]
        clean_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        flat_labels = labels.view(-1)

        if not self._weights_moved and self.loss_fct.weight is not None:
            self.loss_fct.weight = self.loss_fct.weight.to(model.device)
            self._weights_moved = True

        outputs = model(**clean_inputs)
        loss = self.loss_fct(
            outputs.logits.view(-1, outputs.logits.size(-1)), flat_labels)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], *args, **kwargs):
        """覆寫訓練步驟以整合 FGM 對抗式訓練。"""
        if not self.use_fgm or self.fgm is None:
            return super().training_step(model, inputs, *args, **kwargs)

        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]
        clean_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        # 乾淨梯度與對抗梯度各貢獻一半，以 *2 確保最終梯度為兩者的平均值
        scale_factor = self.args.gradient_accumulation_steps * 2

        # 步驟 1：乾淨前向與反向傳遞
        with self.compute_loss_context_manager():
            loss_clean = self.compute_loss(model, inputs)
        self.accelerator.backward(loss_clean / scale_factor)

        # 步驟 2：備份乾淨梯度；透過獨立前向傳遞計算攻擊方向
        # 僅使用加權交叉熵計算梯度，作為 FGM 的攻擊方向
        saved_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                saved_grads[name] = param.grad.detach().clone()
        model.zero_grad(set_to_none=False)

        with self.compute_loss_context_manager():
            outputs_for_fgm = model(**clean_inputs)
            loss_for_fgm = self.loss_fct(
                outputs_for_fgm.logits.view(-1,
                                            outputs_for_fgm.logits.size(-1)),
                labels.view(-1),
            )
        loss_for_fgm.backward()

        # 步驟 3：施加 FGM 擾動；還原乾淨梯度；執行對抗前向與反向傳遞
        self.fgm.attack(epsilon=self.fgm_epsilon)

        for name, param in model.named_parameters():
            param.grad = saved_grads.get(name, None)
        del saved_grads

        with self.compute_loss_context_manager():
            loss_adv = self.compute_loss(model, inputs)
        # 對抗梯度與乾淨梯度取平均
        self.accelerator.backward(loss_adv / scale_factor)

        # 步驟 4：還原 Embedding 權重
        self.fgm.restore()

        total_loss = (loss_clean.detach() + loss_adv.detach()) / 2
        return total_loss / self.args.gradient_accumulation_steps

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
# 詞彙擴充
# =========================================================
def build_fused_tokenizer(base_model_name, extra_tokenizer_name, corpus_texts, min_freq=1):
    """
    以 T-BERT 專有的子詞單元擴充 bert-base-chinese 詞表。
    僅加入 bert-base-chinese 詞表中不存在，且在訓練語料中的子詞頻率大於或等於 min_freq 的詞元。
    新詞的 Embedding 之後會從 T-BERT 預訓練向量初始化。
    """
    base_tok = AutoTokenizer.from_pretrained(base_model_name)
    if extra_tokenizer_name is None:
        return base_tok, [], {}

    extra_tok = AutoTokenizer.from_pretrained(extra_tokenizer_name)
    base_vocab = set(base_tok.get_vocab().keys())
    extra_vocab_dict = extra_tok.get_vocab()
    # 存在於 T-BERT 但不在 bert-base-chinese 的詞元
    candidate_tokens = set(extra_vocab_dict.keys()) - base_vocab

    # 使用 T-BERT 分詞對訓練語料計算候選詞元頻率
    counter = Counter()
    for text in corpus_texts:
        ids = extra_tok.encode(text, add_special_tokens=False)
        toks = extra_tok.convert_ids_to_tokens(ids)
        counter.update(t for t in toks if t in candidate_tokens)

    valid_tokens = {tok for tok, freq in counter.items() if freq >= min_freq}
    final_tokens = sorted(valid_tokens, key=lambda x: (-counter[x], x))
    base_tok.add_tokens(final_tokens)

    # 建立 ID 對照表 {詞元: (base_id, tbert_id)}，用於 Embedding 初始化
    token_id_map = {}
    for tok in final_tokens:
        tbert_id = extra_vocab_dict.get(tok)
        base_id = base_tok.convert_tokens_to_ids(tok)
        if tbert_id is not None and base_id is not None:
            token_id_map[tok] = (base_id, tbert_id)

    print(
        f"    -> [VocabExpansion] T-BERT candidate tokens: {len(candidate_tokens)}")
    print(
        f"    -> [VocabExpansion] tokens retained (freq>={min_freq}): {len(final_tokens)}")

    return base_tok, final_tokens, token_id_map

# =========================================================
# 模型載入與 Embedding 擴充
# =========================================================


def load_model(tokenizer, cfg: CFG, source=None,
               token_id_map: Optional[Dict] = None,
               do_vocab_init: bool = False):
    model_source = source if source else (
        cfg.init_checkpoint if cfg.init_checkpoint else cfg.base_model_name
    )

    config = AutoConfig.from_pretrained(
        model_source,
        num_labels=cfg.num_labels,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
    )

    model = BertForSequenceClassification.from_pretrained(
        model_source, config=config)

    # 擴充 Embedding 矩陣以對應擴充後的詞表大小
    model.resize_token_embeddings(len(tokenizer))

    # 從 T-BERT 預訓練向量初始化新詞 Embedding
    if do_vocab_init and token_id_map and cfg.extra_tokenizer_name:
        _init_new_embeddings_from_tbert(
            model, cfg.extra_tokenizer_name, token_id_map)

    model.to(DEVICE)
    return model


def _init_new_embeddings_from_tbert(model, extra_model_name: str, token_id_map: Dict):
    """將 T-BERT 預訓練向量複製到新增詞槽；複製完成後釋放 T-BERT 模型。"""
    from transformers import AutoModel as _AutoModel
    tbert = _AutoModel.from_pretrained(extra_model_name)

    # 形狀：[tbert_vocab_size, hidden_dim]
    tbert_emb = tbert.embeddings.word_embeddings.weight.data
    # 形狀：[base_vocab_size + num_new, hidden_dim]
    base_emb = model.bert.embeddings.word_embeddings.weight.data

    copied = 0
    with torch.no_grad():
        for tok_str, (base_id, tbert_id) in token_id_map.items():
            if tbert_id < tbert_emb.shape[0] and base_id < base_emb.shape[0]:
                base_emb[base_id] = tbert_emb[tbert_id].detach(
                ).clone().to(base_emb.device)
                copied += 1

    print(
        f"    -> [VocabInit] copied {copied}/{len(token_id_map)} token embeddings from T-BERT")

    del tbert
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 凍結 / 解凍工具函式
# =========================================================
def freeze_all_bert_encoder(model):
    for p in model.bert.encoder.parameters():
        p.requires_grad = False


def unfreeze_embeddings(model, flag=True):
    for p in model.bert.embeddings.parameters():
        p.requires_grad = flag


def unfreeze_head(model):
    if model.bert.pooler is not None:
        for p in model.bert.pooler.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def set_trainable_layers(model, layer_indices: List[int], flag=True):
    for idx in layer_indices:
        for p in model.bert.encoder.layer[idx].parameters():
            p.requires_grad = flag


# =========================================================
# 優化器參數組
# =========================================================
def split_decay_params(named_params):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return decay_params, no_decay_params


def build_optimizer_groups_stage_1(model, cfg: CFG,
                                   new_token_ids: Optional[List[int]] = None):
    """
    建立 Stage 1 的優化器參數組（Encoder 凍結，僅訓練 Embedding、Pooler 與分類器）。

    新詞的梯度保持不變，既有詞的梯度被縮小，以近似較小的更新幅度。

    實作方式：在 AdamW 更新前，以 (emb_lr / new_token_lr) 縮放既有詞梯度，
    近似新舊詞的差異更新幅度。
    """
    groups = []

    if new_token_ids:
        emb_weight = model.bert.embeddings.word_embeddings.weight
        vocab_size = emb_weight.shape[0]
        new_set = set(new_token_ids)
        old_scale = cfg.stage_1_emb_lr / cfg.stage_1_new_token_lr

        # 縮放矩陣：新詞為 1.0（完整學習率），既有詞為 old_scale
        grad_scale = torch.ones(vocab_size, 1, device=emb_weight.device)

        for i in range(vocab_size):
            if i not in new_set:
                grad_scale[i] = old_scale

        def _make_hook(scale):
            def _hook(grad):
                s = scale if scale.device == grad.device else scale.to(
                    grad.device)
                return grad * s
            return _hook

        # Hook 在每次反向傳遞時觸發
        handle = emb_weight.register_hook(_make_hook(grad_scale))
        model._emb_lr_hook_handle = handle

    emb_decay, emb_no_decay = split_decay_params(
        list(model.bert.embeddings.named_parameters()))

    head_named = []
    if model.bert.pooler is not None:
        head_named += list(model.bert.pooler.named_parameters())
    head_named += list(model.classifier.named_parameters())
    head_decay, head_no_decay = split_decay_params(head_named)

    for p_list, lr, wd in [
        (emb_decay,     cfg.stage_1_new_token_lr, cfg.weight_decay),
        (emb_no_decay,  cfg.stage_1_new_token_lr, 0.0),
        (head_decay,    cfg.stage_1_head_lr, cfg.weight_decay),
        (head_no_decay, cfg.stage_1_head_lr, 0.0),
    ]:
        if p_list:
            groups.append({"params": p_list, "lr": lr, "weight_decay": wd})
    return groups


def remove_emb_lr_hook(model):
    handle = getattr(model, '_emb_lr_hook_handle', None)
    if handle is not None:
        handle.remove()
        model._emb_lr_hook_handle = None
        print("    -> [EmbHook] gradient scaling hook removed")


def build_optimizer_groups_progressive(
    model,
    cfg: CFG,
    unfreeze_indices: List[int],
    top_lr: float,
    head_lr: float,
    layerwise_decay: float,
):
    """
    建立 Stage 2 各 Phase 的優化器參數組（逐步解凍）。
    Encoder 層使用由頂而底的幾何衰減學習率（decay^depth）。
    Embedding 學習率在每個 Phase 開始時，依累積解凍層數重新計算（top_lr * decay^total_unfrozen）。
    分類頭使用固定學習率。
    """
    groups = []

    # 累積所有 Phase 的解凍層數（4 → 8 → 12）
    n_unfrozen = len(unfreeze_indices)
    emb_lr = top_lr * (layerwise_decay ** n_unfrozen)
    # Embedding 學習率依累積解凍層數逐 Phase 遞減

    d, nd = split_decay_params(list(model.bert.embeddings.named_parameters()))
    if d:
        groups.append({"params": d, "lr": emb_lr,
                      "weight_decay": cfg.weight_decay})
    if nd:
        groups.append({"params": nd, "lr": emb_lr, "weight_decay": 0.0})

    head_named = []
    if model.bert.pooler is not None:
        head_named += list(model.bert.pooler.named_parameters())
    head_named += list(model.classifier.named_parameters())
    d, nd = split_decay_params(head_named)
    if d:
        groups.append({"params": d, "lr": head_lr,
                      "weight_decay": cfg.weight_decay})
    if nd:
        groups.append({"params": nd, "lr": head_lr, "weight_decay": 0.0})

    sorted_layers = sorted(unfreeze_indices, reverse=True)
    # depth=0 為最頂層的解凍層；學習率向下層幾何衰減
    for depth, idx in enumerate(sorted_layers):
        curr_lr = top_lr * (layerwise_decay ** depth)
        d, nd = split_decay_params(
            list(model.bert.encoder.layer[idx].named_parameters()))
        if d:
            groups.append({"params": d, "lr": curr_lr,
                          "weight_decay": cfg.weight_decay})
        if nd:
            groups.append({"params": nd, "lr": curr_lr, "weight_decay": 0.0})
    return groups


# =========================================================
# 記憶體清理
# =========================================================
def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 訓練與評估核心
# =========================================================
def run_phase(
    phase_name,
    model,
    train_dataset,
    val_dataset,
    tokenizer,
    class_weights,
    optimizer_groups,
    num_epochs,
    output_dir,
    cfg,
):
    """透過 StrongTrainer 執行單一訓練 Phase；將 Phase checkpoint 存至 best_model/。"""
    os.makedirs(output_dir, exist_ok=True)
    # TrainingArguments 需要一個 learning_rate 值，
    # 但實際學習率由 create_optimizer() 中的 optimizer_grouped_parameters 覆寫
    max_lr = max(group["lr"] for group in optimizer_groups)
    has_val = val_dataset is not None

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=max_lr,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch" if has_val else "no",
        logging_strategy="epoch",
        load_best_model_at_end=True if has_val else False,
        metric_for_best_model="macro_f1" if has_val else None,
        greater_is_better=True if has_val else None,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=cfg.early_stopping_patience)] if has_val else []

    # use_fgm=True 且 Phase 在 fgm_enabled_phases 中時啟用 FGM
    phase_use_fgm = cfg.use_fgm and (phase_name in cfg.fgm_enabled_phases)

    tag = []
    if phase_use_fgm:
        tag.append(f"FGM(eps={cfg.fgm_epsilon})")
    if tag:
        print(f"    -> [{' + '.join(tag)}] {phase_name} enabled")

    trainer = StrongTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if has_val else None,
        callbacks=callbacks,
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
        optimizer_grouped_parameters=optimizer_groups,
        use_fgm=phase_use_fgm,
        fgm_epsilon=cfg.fgm_epsilon,
    )

    phase_start = time.time()
    trainer.train()
    elapsed_time = time.time() - phase_start

    # 儲存 Phase checkpoint；CV 時為最佳 macro_f1 模型，重新訓練時（無驗證集）為最後一個 epoch 的權重
    best_path = os.path.join(output_dir, "best_model")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)

    return trainer, elapsed_time


def evaluate_phase(trainer, dataset):
    results = trainer.evaluate(eval_dataset=dataset)
    summary = {
        "accuracy": results.get("eval_accuracy", float("nan")),
        "precision_macro": results.get("eval_precision_macro", float("nan")),
        "recall_macro": results.get("eval_recall_macro", float("nan")),
        "macro_f1": results.get("eval_macro_f1", float("nan")),
        "precision_weighted": results.get("eval_precision_weighted", float("nan")),
        "recall_weighted": results.get("eval_recall_weighted", float("nan")),
        "weighted_f1": results.get("eval_weighted_f1", float("nan")),
    }
    return summary


def get_best_epoch_from_trainer(trainer, fallback_epoch):
    """從 Trainer 狀態取得最佳 checkpoint 對應的 epoch 數。

    優先從 best_model_checkpoint 路徑解析 step 數，再比對 log_history 取得 epoch；
    若解析失敗，則備援為從 log_history 中找 eval_macro_f1 最高的 epoch；
    兩者皆失敗時，回傳 fallback_epoch。
    """
    best_epoch = fallback_epoch

    if trainer.state.best_model_checkpoint:
        try:
            best_step = int(trainer.state.best_model_checkpoint.split("-")[-1])
            for log in trainer.state.log_history:
                if log.get("step") == best_step and "epoch" in log:
                    best_epoch = log["epoch"]
                    break
        except Exception:
            pass

    # 備援：從 log history 找出 eval_macro_f1 最高的 epoch
    if best_epoch == fallback_epoch:
        best_f1 = -1
        for log in trainer.state.log_history:
            if "eval_macro_f1" in log and log["eval_macro_f1"] > best_f1:
                best_f1 = log["eval_macro_f1"]
                best_epoch = log.get("epoch", fallback_epoch)

    return float(best_epoch)


# =========================================================
# 多 Phase 訓練 Pipeline
# =========================================================
def run_training_pipeline(
    run_root,
    train_ds,
    val_ds,
    tokenizer,
    class_weights,
    cfg,
    dynamic_epochs: Optional[Dict[str, float]] = None,
    new_tokens: Optional[List[str]] = None,
    token_id_map: Optional[Dict] = None,
):
    """
    結合 FGM 的兩階段微調 Pipeline：

    Stage 1  — Encoder 完全凍結；僅訓練 Embedding、Pooler 與 Classifier。
                新詞透過梯度 Hook 保留完整梯度；既有詞梯度則按比例縮放，以降低其更新幅度。

    Stage 2  — 逐步由頂而底解凍 Encoder，每個 Phase 解凍 4 層：
                Phase A：新解凍第 8–11 層；訓練第 8–11 層
                Phase B：新解凍第 4–7 層；訓練第 4–11 層
                Phase C：新解凍第 0–3 層；訓練第 0–11 層
               每個 Phase 套用逐層學習率衰減（由頂層起 decay^depth）。

    每個 Phase 從前一 Phase 的 checkpoint 初始化（CV 時為最佳 macro_f1；重新訓練時為最終 epoch 權重）。
    FGM 依 fgm_enabled_phases 的設定，透過 StrongTrainer 套用於指定的訓練階段。
    """

    os.makedirs(run_root, exist_ok=True)
    phase_records = {}

    phases_setup = [
        ("Stage 1", cfg.epochs_stage_1, [], None, build_optimizer_groups_stage_1),

        ("Phase A", cfg.epochs_phase_a, cfg.phase_a_unfreeze_layers,
         (cfg.phase_a_top_layer_lr,
          cfg.phase_a_head_lr, cfg.phase_a_layerwise_decay),
         build_optimizer_groups_progressive),

        ("Phase B", cfg.epochs_phase_b, cfg.phase_b_unfreeze_layers,
         (cfg.phase_b_top_layer_lr,
          cfg.phase_b_head_lr, cfg.phase_b_layerwise_decay),
         build_optimizer_groups_progressive),

        ("Phase C", cfg.epochs_phase_c, cfg.phase_c_unfreeze_layers,
         (cfg.phase_c_top_layer_lr,
          cfg.phase_c_head_lr, cfg.phase_c_layerwise_decay),
         build_optimizer_groups_progressive),
    ]

    _new_tokens = new_tokens or []
    _token_id_map = token_id_map or {}
    new_token_ids = tokenizer.convert_tokens_to_ids(
        _new_tokens) if _new_tokens else []

    active_layers = []  # 追蹤目前已解凍的 Encoder 層
    prev_model_path = None
    prev_phase_dir = None
    final_trainer = None
    # 最後一個 Phase 的名稱，用於決定保留哪個 Trainer
    final_phase_name = phases_setup[-1][0]

    # 依序執行 Stage 1 → Phase A → Phase B → Phase C
    for idx, (phase_name, default_epochs, new_layers, lr_params, optim_builder) in enumerate(phases_setup):
        phase_dir = os.path.join(
            run_root, phase_name.replace(" ", "_").lower())

        if phase_name == "Stage 1" and prev_model_path is None:
            model = load_model(tokenizer, cfg, source=None,
                               token_id_map=_token_id_map, do_vocab_init=True)
        else:
            model = load_model(tokenizer, cfg, source=prev_model_path)

        if cfg.keep_only_final_phase_per_fold and prev_phase_dir is not None:
            shutil.rmtree(prev_phase_dir, ignore_errors=True)

        freeze_all_bert_encoder(model)
        unfreeze_embeddings(model)
        unfreeze_head(model)

        if new_layers:
            active_layers.extend(new_layers)
        if active_layers:
            set_trainable_layers(model, active_layers)

        if idx == 0:
            groups = optim_builder(model, cfg, new_token_ids=new_token_ids)
        else:
            groups = optim_builder(model, cfg, active_layers, *lr_params)

        # 重新訓練時（無驗證集），以 CV 平均最佳 epoch 作為訓練目標
        if val_ds is None:
            target_epochs = int(round(dynamic_epochs[phase_name]))
            print(
                f"    >>> [Retrain Phase] {phase_name}: run {target_epochs} epochs")
        else:
            target_epochs = int(default_epochs)

        trainer, phase_time = run_phase(
            phase_name,
            model,
            train_ds,
            val_ds,
            tokenizer,
            class_weights,
            groups,
            target_epochs,
            phase_dir,
            cfg,
        )

        # 評估並記錄驗證集指標
        if val_ds is not None:
            metrics = evaluate_phase(trainer, val_ds)
            best_epoch = get_best_epoch_from_trainer(
                trainer, target_epochs)
            metrics["best_epoch"] = best_epoch
            phase_records[phase_name] = metrics

        # 將此 checkpoint 傳給下一個 Phase 作為初始化來源
        prev_model_path = os.path.join(phase_dir, "best_model")
        prev_phase_dir = phase_dir
        final_trainer = trainer

        if phase_name != final_phase_name:
            # 刪除中間 Trainer 以釋放記憶體；Phase C 的 Trainer 保留用於錯誤分析
            del trainer, model
            cleanup_memory()

    return phase_records, prev_model_path, final_trainer


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


def append_misclassified_samples(trainer, dataset, raw_texts, raw_labels, fold_no, output_list):
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


def print_cv_summary(fold_phase_metrics, fold_best_epochs, fold_token_counts):
    print("\n" + "=" * 140)
    print("K-FOLD CV SUMMARY (VALIDATION METRICS / BEST EPOCH / NEW TOKENS)")
    print("=" * 140)
    print(
        f"New tokens per fold: {fold_token_counts} | Mean: {np.mean(fold_token_counts):.1f}"
    )
    print("-" * 140)
    print(
        f"{'Phase':<10} | "
        f"{'Accuracy':>10} | "
        f"{'Macro F1':>10} | "
        f"{'Precision(W)':>12} | "
        f"{'Recall(W)':>10} | "
        f"{'Weighted F1':>11} | "
        f"{'Avg Best Epoch':>14}"
    )
    print("-" * 140)

    for phase_name, metric_list in fold_phase_metrics.items():
        epochs = fold_best_epochs[phase_name]

        acc_values = [m["accuracy"] for m in metric_list]
        f1_macro_values = [m["macro_f1"] for m in metric_list]
        p_weighted_values = [m["precision_weighted"] for m in metric_list]
        r_weighted_values = [m["recall_weighted"] for m in metric_list]
        f1_weighted_values = [m["weighted_f1"] for m in metric_list]

        raw_mean_epoch = float(np.mean(epochs))

        print(
            f"{phase_name:<10} | "
            f"{np.mean(acc_values):>10.4f} | "
            f"{np.mean(f1_macro_values):>10.4f} | "
            f"{np.mean(p_weighted_values):>12.4f} | "
            f"{np.mean(r_weighted_values):>10.4f} | "
            f"{np.mean(f1_weighted_values):>11.4f} | "
            f"{raw_mean_epoch:>14.1f}"
        )

    print("-" * 140)

    # 各折最佳 epoch 取平均並四捨五入，作為後續全資料重新訓練的 epoch 目標
    avg_epochs_dict = {}
    for phase_name, epochs in fold_best_epochs.items():
        avg_epochs_dict[phase_name] = int(round(float(np.mean(epochs))))

    print("Retraining epochs per phase:")
    for phase_name, epoch in avg_epochs_dict.items():
        print(f"  {phase_name}: {epoch} epochs")

    return avg_epochs_dict


def evaluate_final_test(model_path, tokenizer, test_ds, cfg,
                        test_texts: Optional[List[str]] = None,
                        test_labels_orig: Optional[List[int]] = None,
                        seed: int = 0):
    eval_start = time.time()
    model = load_model(tokenizer, cfg, source=model_path)
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
    if test_texts is not None and test_labels_orig is not None:
        misclassified_test = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                misclassified_test.append({
                    "Sentence": test_texts[i],
                    "True_Label": int(y_true[i]) + 1,
                    "Predicted_Label": int(y_pred[i]) + 1,
                })
        mis_test_df = pd.DataFrame(misclassified_test)
        mis_test_xlsx = os.path.join(
            cfg.excel_output_dir,
            f"two_stage_fgm_test_misclassified_seed{seed}.xlsx",
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
# 主程式：CV → 平均最佳 epoch → 全資料重新訓練 → 測試集評估
# =========================================================
def main(seed: int = 0):
    set_global_seed(seed)
    print("\n" + "#" * 100)
    print(f"# Multi-seed Experiment: seed = {seed}")
    print("#" * 100)

    total_start = time.time()

    df = pd.read_csv(cfg.csv_path).dropna().reset_index(drop=True)
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
        f"two_stage_fgm_test_samples_seed{seed}.xlsx",
    )
    pd.DataFrame(test_sample_rows).to_excel(test_sample_xlsx, index=False)
    print(
        f"Test sample list saved: {test_sample_xlsx} ({len(test_sample_rows)} samples)")

    # 剩餘 85% 用於分層 5-fold CV
    skf = StratifiedKFold(n_splits=cfg.n_folds,
                          shuffle=True, random_state=SEED)

    # Stage 1 + Phase A–C
    phase_names = ["Stage 1", "Phase A", "Phase B", "Phase C"]
    fold_phase_metrics = {phase: [] for phase in phase_names}
    fold_best_epochs = {phase: [] for phase in phase_names}
    fold_token_counts = []
    misclassified_val_data = []
    fold_phase_table = []

    cv_start = time.time()

    for fold, (train_rel_idx, val_rel_idx) in enumerate(skf.split(trainval_idx, labels_np[trainval_idx]), start=1):
        print("\n" + "=" * 90)
        print(f"Fold {fold} / {cfg.n_folds}")
        print("=" * 90)

        fold_train_idx = trainval_idx[train_rel_idx]
        fold_val_idx = trainval_idx[val_rel_idx]

        # 標籤分佈（還原為原始 1/2/3 編碼）
        t_lbls, t_cnts = np.unique(
            labels_np[fold_train_idx] + 1, return_counts=True)
        v_lbls, v_cnts = np.unique(
            labels_np[fold_val_idx] + 1, return_counts=True)

        train_dist_str = ", ".join(
            [f"Label{lbl}: {cnt}" for lbl, cnt in zip(t_lbls, t_cnts)])
        val_dist_str = ", ".join(
            [f"Label{lbl}: {cnt}" for lbl, cnt in zip(v_lbls, v_cnts)])

        print(
            f"    -> [Data Split] Train: {len(fold_train_idx)} samples [{train_dist_str}]")
        print(
            f"    -> [Data Split] Val:   {len(fold_val_idx):>4} samples [{val_dist_str}]")

        # 僅使用 fold 訓練集建立 Tokenizer，避免驗證集詞彙洩漏
        tokenizer, new_tokens, token_id_map = build_fused_tokenizer(
            cfg.base_model_name,
            cfg.extra_tokenizer_name,
            [texts[i] for i in fold_train_idx],
            min_freq=cfg.min_new_token_freq,
        )
        fold_token_counts.append(len(new_tokens))
        print(f"    -> [Tokenizer] Fold {fold} new tokens: {len(new_tokens)}")

        train_ds = XieHouYuDataset(
            [texts[i] for i in fold_train_idx],
            [labels[i] for i in fold_train_idx],
            tokenizer,
            cfg.max_length,
        )
        val_texts = [texts[i] for i in fold_val_idx]
        val_labels = [labels[i] for i in fold_val_idx]
        val_ds = XieHouYuDataset(
            val_texts, val_labels, tokenizer, cfg.max_length)

        class_weights = torch.tensor(
            compute_class_weight(
                "balanced",
                classes=np.unique(labels_np),
                y=labels_np[fold_train_idx],
            ),
            dtype=torch.float,
        ).to(DEVICE)

        print(
            f"    -> [Class Weights] Fold {fold} class weights: {np.round(class_weights.detach().cpu().numpy(), 4)}")

        fold_root = os.path.join(cfg.output_root, f"seed_{seed}_fold_{fold}")
        records, _, final_phase_trainer = run_training_pipeline(
            fold_root,
            train_ds,
            val_ds,
            tokenizer,
            class_weights,
            cfg,
            dynamic_epochs=None,
            new_tokens=new_tokens,
            token_id_map=token_id_map,
        )

        for phase_name, metrics in records.items():
            fold_phase_metrics[phase_name].append(metrics)
            fold_best_epochs[phase_name].append(metrics["best_epoch"])

            fold_phase_table.append({
                "Fold": fold,
                "Phase": phase_name,
                "Accuracy": metrics["accuracy"],
                "Precision_macro": metrics["precision_macro"],
                "Recall_macro": metrics["recall_macro"],
                "F1_macro": metrics["macro_f1"],
                "Precision_weighted": metrics["precision_weighted"],
                "Recall_weighted": metrics["recall_weighted"],
                "F1_weighted": metrics["weighted_f1"],
                "best_epoch": metrics["best_epoch"],
            })

        final_phase_metrics = records["Phase C"]

        print(
            f"[Fold {fold} Result] "
            f"Accuracy={final_phase_metrics['accuracy']:.4f} | "
            f"F1-Macro={final_phase_metrics['macro_f1']:.4f} | "
            f"F1-Weighted={final_phase_metrics['weighted_f1']:.4f}"
        )

        # 收集 Phase C 驗證集分類錯誤樣本，用於錯誤分析
        append_misclassified_samples(
            final_phase_trainer,
            val_ds,
            val_texts,
            val_labels,
            fold,
            misclassified_val_data,
        )

        del final_phase_trainer, train_ds, val_ds
        cleanup_memory()

    print("\n" + "=" * 90)
    cv_elapsed_min = (time.time() - cv_start) / 60
    print(f"CV elapsed: {cv_elapsed_min:.2f} min")
    print("=" * 90)

    # 匯出各折 × 各 Phase 的完整指標
    fold_phase_df = pd.DataFrame(fold_phase_table)

    phase_order = {phase: i for i, phase in enumerate(phase_names, start=1)}
    fold_phase_df["Phase_Order"] = fold_phase_df["Phase"].map(phase_order)
    fold_phase_df = fold_phase_df.sort_values(
        ["Fold", "Phase_Order"]).drop(columns=["Phase_Order"])

    output_cols = [
        "Fold",
        "Phase",
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "Precision_weighted",
        "Recall_weighted",
        "F1_weighted",
        "best_epoch",
    ]
    fold_phase_df = fold_phase_df[output_cols]

    metric_cols = [
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "Precision_weighted",
        "Recall_weighted",
        "F1_weighted",
        "best_epoch",
    ]
    fold_phase_df[metric_cols] = fold_phase_df[metric_cols].round(4)

    fold_phase_xlsx_name = os.path.join(

        cfg.excel_output_dir,

        f"two_stage_fgm_cv_each_fold_each_phase_metrics_seed{seed}.xlsx",

    )
    fold_phase_df.to_excel(fold_phase_xlsx_name, index=False)

    print("Full fold × phase metrics saved:")
    print(f"{fold_phase_xlsx_name}")

    phase_c_df = fold_phase_df[fold_phase_df["Phase"] == "Phase C"].copy()
    phase_c_df = phase_c_df[[
        "Fold",
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "Precision_weighted",
        "Recall_weighted",
        "F1_weighted",
        "best_epoch",
    ]]

    phase_c_xlsx_name = os.path.join(

        cfg.excel_output_dir,

        f"two_stage_fgm_phase_c_each_fold_metrics_seed{seed}.xlsx",

    )
    phase_c_df.to_excel(phase_c_xlsx_name, index=False)

    print("Phase C per-fold metrics saved:")
    print(f"{phase_c_xlsx_name}")

    # 匯出驗證集分類錯誤報告
    mis_val_df = pd.DataFrame(misclassified_val_data)
    mis_xlsx_name = os.path.join(
        cfg.excel_output_dir,
        f"two_stage_fgm_phase_c_best_misclassified_val_seed{seed}.xlsx",
    )
    mis_val_df.to_excel(mis_xlsx_name, index=False)
    print(
        f"Phase C validation misclassification report saved: {mis_xlsx_name} ({len(mis_val_df)} samples)")

    # 從各折 CV 平均最佳 epoch 推導各 Phase 的重新訓練 epoch 目標
    dynamic_epochs = print_cv_summary(
        fold_phase_metrics, fold_best_epochs, fold_token_counts)

    # 依據 CV 推導的各 Phase epoch 數，
    # 使用完整的訓練與驗證資料重新訓練模型
    print("\n" + "=" * 90)
    print("RETRAINING TWO-STAGE MODEL ON 100% OF TRAIN-VAL DATA")
    print("=" * 90)

    retrain_tokenizer, retrain_tokens, retrain_token_id_map = build_fused_tokenizer(
        cfg.base_model_name,
        cfg.extra_tokenizer_name,
        [texts[i] for i in trainval_idx],
        min_freq=cfg.min_new_token_freq,
    )
    print(f"Retrain model new tokens: {len(retrain_tokens)}")

    trainval_ds = XieHouYuDataset(
        [texts[i] for i in trainval_idx],
        [labels[i] for i in trainval_idx],
        retrain_tokenizer,
        cfg.max_length,
    )
    test_ds = XieHouYuDataset(
        [texts[i] for i in test_idx],
        [labels[i] for i in test_idx],
        retrain_tokenizer,
        cfg.max_length,
    )

    # 使用完整的訓練與驗證資料計算類別權重
    retrain_weights = compute_class_weight(
        "balanced",
        classes=np.unique(labels_np),
        y=labels_np[trainval_idx],
    )

    class_weights = torch.tensor(retrain_weights, dtype=torch.float).to(DEVICE)

    print(
        f"    -> [Retrain Class Weights] class weights: {np.round(retrain_weights, 4)}")

    retrain_root = os.path.join(
        cfg.output_root, f"seed_{seed}_retrain_two_stage_fgm_model")
    retrain_start = time.time()
    _, final_model_path, retrain_trainer = run_training_pipeline(
        retrain_root,
        trainval_ds,
        None,
        retrain_tokenizer,
        class_weights,
        cfg,
        dynamic_epochs=dynamic_epochs,
        new_tokens=retrain_tokens,
        token_id_map=retrain_token_id_map,
    )
    retraining_elapsed_min = (time.time() - retrain_start) / 60
    print(f"RETRAINING elapsed: {retraining_elapsed_min:.2f} min")

    # 計算 5-fold 分層交叉驗證與完整訓練驗證集重新訓練的總時間；
    # 不包含最終測試集推論與評估時間。
    cv_retraining_time_min = cv_elapsed_min + retraining_elapsed_min
    print(
        f"CV + RETRAINING TIME:"
        f"{cv_retraining_time_min:.2f} min"
    )

    # 重新訓練模型已儲存至 final_model_path；測試前釋放 Trainer 與 GPU 模型記憶體。
    del retrain_trainer
    cleanup_memory()

    # 在保留的 15% 測試集上進行最終評估
    test_texts_raw = [texts[i] for i in test_idx]
    test_labels_raw = [labels_np[i] + 1 for i in test_idx]
    _, test_metrics = evaluate_final_test(
        final_model_path, retrain_tokenizer, test_ds, cfg,
        test_texts=test_texts_raw,
        test_labels_orig=test_labels_raw,
        seed=seed,
    )

    total_time_min = (time.time() - total_start) / 60
    print("\n" + "=" * 90)
    print(f"Total elapsed: {total_time_min:.2f} min")
    print("=" * 90)

    # 彙整此 seed 的結果
    phase_c_fold_metrics = fold_phase_metrics["Phase C"]
    val_summary = {
        "val_phase_c_accuracy_mean": float(np.mean([m["accuracy"] for m in phase_c_fold_metrics])),
        "val_phase_c_accuracy_std":  float(np.std([m["accuracy"] for m in phase_c_fold_metrics])),
        "val_phase_c_macro_f1_mean": float(np.mean([m["macro_f1"] for m in phase_c_fold_metrics])),
        "val_phase_c_macro_f1_std":  float(np.std([m["macro_f1"] for m in phase_c_fold_metrics])),
        "val_phase_c_weighted_f1_mean": float(np.mean([m["weighted_f1"] for m in phase_c_fold_metrics])),
        "val_phase_c_weighted_f1_std":  float(np.std([m["weighted_f1"] for m in phase_c_fold_metrics])),
    }

    seed_result = {
        "seed": seed,
        **val_summary,
        **test_metrics,
        "cv_retraining_minutes": cv_retraining_time_min,
        "total_minutes": total_time_min,
    }
    return seed_result


# =========================================================
# 多 Seed 實驗
# =========================================================
def run_multi_seed_experiment(seeds=(0, 1, 42, 123, 1234)):

    print("\n" + "*" * 100)
    print(f"* Multi-seed experiment: {len(seeds)} seeds = {list(seeds)}")
    print("*" * 100)
    multi_start = time.time()
    seed_results = []

    for seed in seeds:
        result = main(seed=seed)
        seed_results.append(result)
        # 每個 seed 結束後儲存部分結果，避免中斷遺失
        _save_multi_seed_report(seed_results, suffix="partial")

    total_min = (time.time() - multi_start) / 60
    print("\n" + "*" * 100)
    print(
        f"* Multi-seed experiment complete. Total elapsed: {total_min:.2f} min")
    print("*" * 100)

    _save_multi_seed_report(seed_results, suffix="final")
    _print_multi_seed_summary(seed_results)
    return seed_results


def _save_multi_seed_report(seed_results, suffix="final"):
    df = pd.DataFrame(seed_results)

    numeric_cols = [c for c in df.columns if c != "seed"]
    overall_mean = df[numeric_cols].mean().to_dict()
    overall_std = df[numeric_cols].std().to_dict()
    overall_mean["seed"] = "MEAN"
    overall_std["seed"] = "STD"

    df_out = pd.concat([
        df,
        pd.DataFrame([overall_mean]),
        pd.DataFrame([overall_std]),
    ], ignore_index=True)

    df_out = df_out.round(4)
    out_name = os.path.join(
        cfg.excel_output_dir,
        f"two_stage_fgm_multi_seed_report_{suffix}.xlsx",
    )
    df_out.to_excel(out_name, index=False)
    print(f"[Multi-seed report] saved: {out_name}")


def _print_multi_seed_summary(seed_results):
    df = pd.DataFrame(seed_results)
    print("\n" + "=" * 100)
    print("MULTI-SEED EXPERIMENT SUMMARY")
    print("=" * 100)
    print(
        f"{'Seed':<8}"
        f"{'Val Acc':>10}"
        f"{'Val MacroF1':>14}"
        f"{'Val WgtF1':>12}"
        f"{'Test Acc':>11}"
        f"{'Test MacroF1':>15}"
        f"{'Test WgtF1':>13}"
        f"{'CV+Retrain(min)':>17}"
        f"{'Total(min)':>12}"
    )
    print("-" * 100)
    for _, row in df.iterrows():
        print(f"{int(row['seed']):<8}"
              f"{row['val_phase_c_accuracy_mean']:>10.4f}"
              f"{row['val_phase_c_macro_f1_mean']:>14.4f}"
              f"{row['val_phase_c_weighted_f1_mean']:>12.4f}"
              f"{row['test_accuracy']:>11.4f}"
              f"{row['test_macro_f1']:>15.4f}"
              f"{row['test_weighted_f1']:>13.4f}"
              f"{row['cv_retraining_minutes']:>17.2f}"
              f"{row['total_minutes']:>12.2f}")
    print("-" * 100)
    # 跨 seed 統計
    for metric in ["val_phase_c_macro_f1_mean", "test_macro_f1",
                   "val_phase_c_accuracy_mean", "test_accuracy",
                   "val_phase_c_weighted_f1_mean", "test_weighted_f1"]:
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"  Cross-seed {metric:<35}: {mean:.4f} ± {std:.4f}")
    print("=" * 100)


if __name__ == "__main__":
    run_multi_seed_experiment(seeds=(0, 1, 42, 123, 1234))
