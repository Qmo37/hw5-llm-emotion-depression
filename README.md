# HW5 — LLM 微調：情緒分類與憂鬱症風險監測

**LoRA / Zero-shot / Few-shot**

## 專案概述

本專案實作大型語言模型（LLM）的情緒分類與憂鬱症風險監測系統，比較三種方法的效能：
- **Zero-shot**：無需訓練的直接推論
- **Few-shot**：利用少量示例進行推論
- **LoRA 微調**：參數高效的微調方法

## 作業目標

1. 使用 HuggingFace Datasets 載入大型開源情緒資料集
2. 建立從 Emotion → Depression-risk 的轉換規則（risk mapping）
3. 比較 Zero-shot、Few-shot、LoRA 微調三種方法
4. 完成情緒分類與風險分類（低／中／高）
5. 訓練並評估模型效能（F1、AUROC、PR-AUC、Confusion Matrix）
6. 完成基礎的憂鬱症風險監測視覺化（走勢圖 + 熱圖）
7. 撰寫技術報告

## 資料集

**Emotion Dataset** (Saravia et al., 2018)
- 來源：[HuggingFace - dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- 情緒標籤：joy, love, surprise, anger, fear, sadness
- 資料筆數：
  - Training: 16,000
  - Validation: 2,000
  - Test: 2,000

## 風險映射規則

| 情緒類別 | 風險等級 |
|---------|---------|
| joy / love / surprise | low_risk (0) |
| anger / fear | mid_risk (1) |
| sadness | high_risk (2) |

## 模型設定

- **基礎模型**：TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **量化方式**：4-bit NF4 quantization
- **最大序列長度**：128 tokens
- **LoRA 參數**：
  - rank (r): 16
  - alpha: 32
  - dropout: 0.05
  - target modules: q_proj, k_proj, v_proj, o_proj

## 訓練參數

- **訓練回合數**：3 epochs
- **批次大小**：4 (per device)
- **梯度累積步數**：4
- **學習率**：2e-4
- **優化器**：paged_adamw_8bit
- **學習率調度器**：cosine with warmup

## 檔案結構

```
hw5-llm-emotion-depression/
├── HW5_LLM_Emotion_Depression.ipynb  # 主要 Colab notebook
├── README.md                          # 專案說明文件
├── Technical_Report.md                # 技術報告（需撰寫）
└── outputs/                           # 輸出資料夾（執行後生成）
    ├── lora_emotion_final/            # 訓練完成的 LoRA 模型
    ├── method_comparison.png          # 方法比較圖
    ├── confusion_matrices.png         # 混淆矩陣
    ├── risk_trend.png                 # 風險趨勢圖
    ├── risk_heatmap.png               # 風險熱圖
    ├── risk_comprehensive.png         # 綜合風險分析圖
    └── HW5_Summary_Report.txt         # 結果摘要報告
```

## 使用方式

### 1. 在 Google Colab 中執行

1. 開啟 [Google Colab](https://colab.research.google.com/)
2. 上傳 `HW5_LLM_Emotion_Depression.ipynb`
3. 確保已啟用 GPU runtime：
   - Runtime → Change runtime type → Hardware accelerator → GPU
4. 依序執行所有 cells

### 2. 環境需求

```bash
# 主要套件
transformers
datasets
accelerate
peft
bitsandbytes
sentencepiece

# 數據分析與視覺化
scikit-learn
matplotlib
seaborn
pandas
numpy
tqdm
```

### 3. 執行流程

1. **環境設置**：安裝必要套件
2. **資料載入**：從 HuggingFace 載入 Emotion 資料集
3. **風險映射**：建立情緒到風險的對應關係
4. **Zero-shot 推論**：使用未經訓練的模型進行推論
5. **Few-shot 推論**：加入示例後進行推論
6. **LoRA 微調**：訓練 LoRA 適配器
7. **評估比較**：計算各種指標並比較三種方法
8. **風險視覺化**：生成風險監測圖表
9. **報告產出**：生成摘要報告

## 評估指標

### 情緒分類指標
- F1 Score (Macro)
- F1 Score (Weighted)
- Confusion Matrix

### 風險分類指標
- F1 Score (Macro)
- F1 Score (Weighted)
- AUROC (Area Under ROC Curve)
- PR-AUC (Precision-Recall AUC)

## 視覺化輸出

1. **方法比較圖**：比較三種方法的 F1 分數
2. **混淆矩陣**：情緒分類與風險分類的混淆矩陣（6 張圖）
3. **風險趨勢圖**：高風險機率隨樣本索引的變化
4. **風險熱圖**：使用滾動窗口的風險濃度熱圖
5. **綜合風險圖**：所有風險等級的分布與趨勢

## 重要限制與注意事項

### 模型限制
- 僅基於文本進行情緒檢測，未考慮其他模態（語音、影像等）
- 風險映射規則簡化，未經臨床驗證
- 模型可能存在訓練資料的偏差

### 倫理考量
- **不應作為臨床診斷工具**使用
- 需要心理健康專業人員的驗證
- 實際應用需考慮隱私保護與倫理問題
- 可能存在對特定群體的偏見

### 技術限制
- 受限於 Colab 免費版的 GPU 記憶體
- 評估樣本數量有限（為加快執行速度）
- 簡化的機率估計（硬分類轉換為 one-hot）
