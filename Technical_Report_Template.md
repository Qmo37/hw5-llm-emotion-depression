# HW5 技術報告：LLM 微調於情緒分類與憂鬱症風險監測

**學生姓名**：[請填寫]  
**學號**：[請填寫]  
**日期**：2025-11-07

---

## 摘要 (Abstract)

[150-200 字摘要，包含：研究目的、使用方法、主要結果、結論]

本研究探討大型語言模型在情緒分類與憂鬱症風險監測中的應用。我們使用 HuggingFace Emotion 資料集，比較了 Zero-shot、Few-shot 和 LoRA 微調三種方法的效能。實驗結果顯示...

---

## 1. 引言 (Introduction)

### 1.1 研究背景與動機

社群媒體已成為人們表達情緒與心理狀態的重要平台。透過分析社群文本中的情緒訊號（如焦慮、悲傷、無力感等），可以作為心理健康趨勢的代理指標（proxy indicator）。隨著大型語言模型（Large Language Models, LLMs）的快速發展，我們有機會運用這些先進技術來建立更有效的情緒分析與風險監測系統。

### 1.2 研究目標

本研究的主要目標包括：
1. 實作並比較三種 LLM 推論方法（Zero-shot、Few-shot、LoRA 微調）
2. 建立情緒分類到憂鬱症風險等級的映射機制
3. 開發風險監測視覺化系統
4. 評估各方法在情緒分類與風險預測上的效能

### 1.3 研究貢獻

- 提供完整的 LLM 微調流程實作
- 比較不同推論策略的效能差異
- 展示參數高效微調（PEFT）在資源受限環境下的應用
- 建立憂鬱症風險監測的視覺化工具

---

## 2. 資料集說明 (Dataset Description)

### 2.1 資料來源

- **資料集名稱**：Emotion Dataset (Saravia et al., 2018)
- **來源**：HuggingFace Datasets - `dair-ai/emotion`
- **資料特性**：英文短文本情緒分類資料集

### 2.2 資料結構

| 分割 (Split) | 樣本數 | 說明 |
|-------------|--------|------|
| Training | 16,000 | 訓練集 |
| Validation | 2,000 | 驗證集 |
| Test | 2,000 | 測試集 |

### 2.3 情緒標籤

資料集包含 6 種情緒標籤：
1. **sadness** (悲傷)
2. **joy** (快樂)
3. **love** (愛)
4. **anger** (憤怒)
5. **fear** (恐懼)
6. **surprise** (驚訝)

### 2.4 資料分布分析

[請在此插入資料分布圖表，分析各情緒類別的樣本數量分布]

**觀察結果**：
- [描述資料分布特性]
- [是否存在類別不平衡問題]
- [對模型訓練的潛在影響]

---

## 3. 風險映射規則 (Risk Mapping)

### 3.1 映射設計理念

基於心理學文獻與情緒理論，我們將六種情緒對應到三個憂鬱症風險等級：

| 情緒類別 | 風險等級 | 風險標籤 | 理論依據 |
|---------|---------|---------|----------|
| joy, love, surprise | 0 | low_risk | 正向情緒，通常與良好心理狀態相關 |
| anger, fear | 1 | mid_risk | 負向情緒，可能為壓力或焦慮的徵兆 |
| sadness | 2 | high_risk | 持續悲傷為憂鬱症的核心症狀 |

### 3.2 映射函數實作

```python
emotion_to_risk = {
    0: 2,  # sadness -> high_risk
    1: 0,  # joy -> low_risk
    2: 0,  # love -> low_risk
    3: 1,  # anger -> mid_risk
    4: 1,  # fear -> mid_risk
    5: 0   # surprise -> low_risk
}
```

### 3.3 風險分布

[請在此插入風險分布圖表]

**統計摘要**：
- Low risk: [樣本數] ([百分比])
- Mid risk: [樣本數] ([百分比])
- High risk: [樣本數] ([百分比])

---

## 4. 模型選擇與設定 (Model Configuration)

### 4.1 基礎模型選擇

**選用模型**：TinyLlama/TinyLlama-1.1B-Chat-v1.0

**選擇理由**：
1. **模型大小適中**：1.1B 參數，適合 Colab 免費版 GPU 記憶體限制
2. **對話優化**：Chat 版本已針對指令跟隨進行優化
3. **開源可用**：完全開源，可自由使用與修改
4. **效能平衡**：在小型模型中具有競爭力的效能

### 4.2 量化配置

為了在有限的 GPU 記憶體下運行模型，我們使用 4-bit 量化：

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 量化類型
    bnb_4bit_compute_dtype=torch.float16, # 計算使用 FP16
    bnb_4bit_use_double_quant=True,      # 雙重量化以進一步節省記憶體
)
```

**量化效益**：
- 記憶體需求降低約 75%
- 推論速度提升
- 效能損失在可接受範圍內

### 4.3 Tokenizer 設定

- **Padding token**：設定為 `eos_token`
- **Padding side**：右側填充（right）
- **最大長度**：128 tokens
- **截斷策略**：超過最大長度則截斷

---

## 5. 方法實作 (Methods Implementation)

### 5.1 Zero-shot 推論

**定義**：不提供任何訓練範例，僅透過精心設計的提示詞（prompt）引導模型進行分類。

**Prompt 設計**：
```
<|system|>
You are an emotion classifier. Classify the following text into one of these emotions: 
sadness, joy, love, anger, fear, surprise.
Respond with only the emotion label, nothing else.
<|user|>
Text: {text}
Emotion:
<|assistant|>
```

**優點**：
- 無需訓練，快速部署
- 靈活性高，易於調整

**缺點**：
- 效能受限於預訓練知識
- 對提示詞設計敏感

**實驗設定**：
- 評估樣本數：200（隨機抽樣）
- Temperature：0.1（低溫度以提高穩定性）
- Max new tokens：10

### 5.2 Few-shot 推論

**定義**：在提示詞中加入 3-5 個示範範例，幫助模型理解任務格式與期望輸出。

**Prompt 設計**：
```
<|system|>
You are an emotion classifier. Classify text into one of these emotions: 
sadness, joy, love, anger, fear, surprise.
Here are some examples:
<|user|>
Text: {example_text_1}
Emotion:
<|assistant|>
{example_emotion_1}
...
[3-5 examples]
...
<|user|>
Text: {test_text}
Emotion:
<|assistant|>
```

**範例選擇策略**：
- 每種情緒至少一個範例
- 隨機選擇以避免偏差
- 文本長度適中，具代表性

**優點**：
- 效能優於 Zero-shot
- 仍無需模型訓練

**缺點**：
- Prompt 長度增加
- 範例選擇影響效能

**實驗設定**：
- 評估樣本數：200
- 示範範例數：5
- Max length：512（以容納範例）

### 5.3 LoRA 微調

**定義**：Low-Rank Adaptation (LoRA) 是一種參數高效微調方法，僅訓練少量額外參數。

#### 5.3.1 LoRA 原理

LoRA 透過在預訓練模型的權重矩陣上添加低秩分解矩陣來實現微調：

$$W' = W + BA$$

其中：
- $W$：預訓練權重（凍結）
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$：可訓練的低秩矩陣
- $r$：秩（rank），通常 $r \ll \min(d, k)$

#### 5.3.2 LoRA 配置

```python
LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # LoRA scaling factor
    lora_dropout=0.05,       # Dropout rate
    bias="none",             # 不訓練 bias
    task_type="CAUSAL_LM",   # 因果語言模型任務
    target_modules=[         # 目標模組
        "q_proj",            # Query projection
        "k_proj",            # Key projection  
        "v_proj",            # Value projection
        "o_proj"             # Output projection
    ]
)
```

**參數說明**：
- **r (rank)**：控制可訓練參數數量，越大則表達能力越強但參數越多
- **lora_alpha**：縮放因子，實際更新為 $(alpha/r) \times BA$
- **target_modules**：選擇 Transformer 注意力層進行微調

#### 5.3.3 訓練參數

| 參數 | 值 | 說明 |
|-----|-----|------|
| Epochs | 3 | 訓練回合數 |
| Batch size (per device) | 4 | 單設備批次大小 |
| Gradient accumulation steps | 4 | 梯度累積步數（有效批次=16） |
| Learning rate | 2e-4 | 學習率 |
| Optimizer | paged_adamw_8bit | 8-bit 分頁 AdamW |
| LR scheduler | cosine | 餘弦退火調度器 |
| Warmup ratio | 0.05 | 預熱步數比例 |
| FP16 | True | 混合精度訓練 |

#### 5.3.4 訓練策略

**資料格式化**：
```python
def format_instruction(sample):
    return f"""<|system|>
You are an emotion classifier...
<|user|>
Text: {sample['text']}
Emotion:
<|assistant|>
{sample['emotion_name']}</s>"""
```

**優點**：
- 參數效率高（僅訓練 0.1-1% 參數）
- 記憶體需求低
- 訓練速度快
- 效能通常最佳

**缺點**：
- 需要訓練時間
- 需要訓練資料

**可訓練參數統計**：
```
[請在執行後填寫]
Total parameters: [X]
Trainable parameters: [Y]
Trainable %: [Z]
```

---

## 6. 訓練流程 (Training Pipeline)

### 6.1 資料預處理

1. **載入資料集**：從 HuggingFace 下載 Emotion 資料集
2. **風險標註**：根據映射規則添加風險標籤
3. **格式化**：將資料格式化為模型可接受的提示詞格式
4. **Tokenization**：使用 tokenizer 將文本轉換為 token IDs

### 6.2 訓練過程

[請在此插入訓練曲線圖，包含 training loss 和 validation loss]

**訓練觀察**：
- [描述 loss 下降趨勢]
- [是否出現過擬合]
- [最佳 checkpoint 選擇]

### 6.3 超參數調整

[如有進行超參數搜索，請記錄嘗試過的配置與結果]

---

## 7. 評估指標與結果 (Evaluation Metrics & Results)

### 7.1 評估指標定義

#### 7.1.1 F1 Score

**Macro F1**：各類別 F1 的平均值（未加權）
$$F1_{macro} = \frac{1}{C} \sum_{i=1}^{C} F1_i$$

**Weighted F1**：根據各類別樣本數加權的 F1 平均值
$$F1_{weighted} = \sum_{i=1}^{C} w_i \cdot F1_i$$

其中 $w_i = \frac{n_i}{N}$（類別 $i$ 的樣本比例）

#### 7.1.2 AUROC (Area Under ROC Curve)

衡量模型在不同閾值下的分類能力，取值範圍 [0, 1]，越高越好。

#### 7.1.3 PR-AUC (Precision-Recall AUC)

在類別不平衡情況下比 AUROC 更具參考價值。

#### 7.1.4 Confusion Matrix

視覺化真實標籤與預測標籤的對應關係。

### 7.2 實驗結果

#### 7.2.1 整體效能比較

[請在此插入結果表格]

| Method | Emotion F1 (Macro) | Emotion F1 (Weighted) | Risk F1 (Macro) | Risk F1 (Weighted) | AUROC | PR-AUC |
|--------|-------------------|---------------------|----------------|-------------------|-------|--------|
| Zero-shot | [填寫] | [填寫] | [填寫] | [填寫] | [填寫] | [填寫] |
| Few-shot | [填寫] | [填寫] | [填寫] | [填寫] | [填寫] | [填寫] |
| LoRA | [填寫] | [填寫] | [填寫] | [填寫] | [填寫] | [填寫] |

[請在此插入方法比較圖]

#### 7.2.2 混淆矩陣分析

[請在此插入 6 張混淆矩陣（3 個方法 × 2 個任務）]

**Zero-shot 分析**：
- [分析哪些情緒容易混淆]
- [錯誤模式]

**Few-shot 分析**：
- [相較 Zero-shot 的改進]
- [仍存在的問題]

**LoRA 分析**：
- [最佳表現的類別]
- [仍具挑戰性的類別]

#### 7.2.3 各情緒類別詳細報告

[請在此插入 classification report]

```
Zero-shot Classification Report:
              precision    recall  f1-score   support

     sadness       [X]       [X]       [X]       [X]
         joy       [X]       [X]       [X]       [X]
        love       [X]       [X]       [X]       [X]
       anger       [X]       [X]       [X]       [X]
        fear       [X]       [X]       [X]       [X]
    surprise       [X]       [X]       [X]       [X]

    accuracy                           [X]       [X]
   macro avg       [X]       [X]       [X]       [X]
weighted avg       [X]       [X]       [X]       [X]
```

### 7.3 結果討論

#### 7.3.1 方法比較

**效能排序**：
1. [最佳方法] - [原因分析]
2. [次佳方法] - [原因分析]
3. [第三方法] - [原因分析]

**改進幅度**：
- LoRA vs Zero-shot：Emotion F1 提升 [X]%
- LoRA vs Few-shot：Emotion F1 提升 [X]%

#### 7.3.2 錯誤分析

**常見錯誤類型**：
1. [情緒 A] 誤判為 [情緒 B]：[可能原因]
2. [情緒 C] 誤判為 [情緒 D]：[可能原因]

**案例分析**：
```
Text: [錯誤預測的範例文本]
True: [真實標籤]
Predicted: [預測標籤]
Analysis: [為何會誤判]
```

---

## 8. 風險監測視覺化 (Risk Monitoring Visualization)

### 8.1 高風險趨勢圖

[請在此插入 risk_trend.png]

**圖表說明**：
- 橫軸：樣本索引（時間序列代理）
- 縱軸：P(high_risk) 機率
- 紅色半透明線：原始機率值
- 深紅色實線：滾動平均（窗口大小=50）
- 橙色虛線：警示閾值（0.3）

**觀察結果**：
- 平均高風險機率：[X]
- 高於閾值的樣本數：[Y] ([Z]%)
- 風險高峰區段：[描述]

### 8.2 風險濃度熱圖

[請在此插入 risk_heatmap.png]

**圖表說明**：
- 使用滾動窗口（大小=50，步長=10）
- 顏色深淺代表風險濃度
- 可識別高風險集中時段

**觀察結果**：
- 高風險集中窗口：[編號]
- 風險分布模式：[描述]

### 8.3 綜合風險分析

[請在此插入 risk_comprehensive.png]

**左圖 - 風險分布**：
- Low risk: [數量] ([百分比])
- Mid risk: [數量] ([百分比])
- High risk: [數量] ([百分比])

**右圖 - 多層級趨勢**：
- [描述三條曲線的關係]
- [風險等級之間的轉換模式]

### 8.4 實際應用建議

**警示系統設計**：
1. **即時監測**：當 P(high_risk) > 0.3 時發出警示
2. **趨勢分析**：追蹤滾動平均的上升/下降趨勢
3. **持續性評估**：高風險狀態持續 N 天以上時升級警示

**限制與注意事項**：
- 此系統僅為輔助工具，不能取代專業診斷
- 需結合其他資訊（行為、生理指標等）
- 隱私保護是首要考量

---

## 9. 模型限制、偏差與潛在風險 (Limitations, Biases, and Risks)

### 9.1 模型限制

#### 9.1.1 技術限制
1. **單模態限制**：僅處理文本，忽略語音語調、肢體語言等非語言訊號
2. **上下文長度**：最大 128 tokens，無法處理長文本或多輪對話
3. **語言限制**：模型主要訓練於英文，對其他語言效果未知
4. **即時性**：推論速度可能無法滿足大規模即時監測需求

#### 9.1.2 方法論限制
1. **風險映射簡化**：將情緒直接映射到風險等級過於簡化
2. **缺乏時序建模**：未考慮情緒隨時間的變化模式
3. **評估樣本有限**：為加快實驗僅使用部分測試集
4. **硬分類限制**：未輸出機率分布，僅有確定性預測

### 9.2 資料偏差

#### 9.2.1 訓練資料偏差
1. **人口統計偏差**：資料集可能不均衡地代表不同年齡、性別、文化背景
2. **平台偏差**：文本來源若集中於特定社群平台，可能有表達風格偏差
3. **標註偏差**：人工標註可能存在主觀性與不一致性
4. **時間偏差**：資料收集時間點可能影響情緒表達方式

#### 9.2.2 類別不平衡
- [分析各情緒類別的樣本數差異]
- [對模型訓練的影響]
- [採取的緩解措施]

### 9.3 倫理風險

#### 9.3.1 隱私風險
1. **敏感資訊**：情緒與心理健康資訊屬於敏感個人資料
2. **去識別化挑戰**：文本內容可能包含可識別個人的資訊
3. **資料外洩**：模型可能記憶訓練資料中的個人資訊

#### 9.3.2 誤用風險
1. **過度解讀**：將模型預測當作臨床診斷依據
2. **歧視風險**：基於風險預測對個人採取不當措施
3. **監控濫用**：未經同意的心理狀態監控侵犯隱私

#### 9.3.3 心理影響
1. **標籤效應**：被標記為高風險可能造成自我實現預言
2. **過度依賴**：依賴自動化系統而忽視人際支持
3. **錯誤警示**：假陽性可能造成不必要的焦慮

### 9.4 臨床有效性

**重要聲明**：
- ✗ 未經臨床驗證
- ✗ 未與專業診斷標準對齊（如 DSM-5）
- ✗ 未考慮持續時間、嚴重程度等診斷要素
- ✗ 無法取代專業心理健康評估

**建議使用方式**：
- ✓ 作為初步篩檢工具
- ✓ 輔助專業人員評估
- ✓ 研究與教育目的

### 9.5 緩解策略

#### 9.5.1 技術面
1. **模型改進**：使用更大規模、多語言模型
2. **機率輸出**：提供不確定性估計
3. **可解釋性**：增加注意力視覺化與特徵重要性分析
4. **持續評估**：定期更新與驗證模型效能

#### 9.5.2 流程面
1. **人工審核**：高風險預測需人工複核
2. **多維度評估**：結合多種資訊來源
3. **專業諮詢**：與心理健康專業人員合作
4. **使用者教育**：明確說明系統限制與正確使用方式

#### 9.5.3 倫理面
1. **知情同意**：使用前取得明確同意
2. **資料保護**：嚴格的資料加密與存取控制
3. **透明度**：公開模型架構、訓練資料、效能指標
4. **問責機制**：建立錯誤回報與修正流程

---

## 10. 結論與未來工作 (Conclusion and Future Work)

### 10.1 研究總結

本研究成功實作並比較了三種 LLM 推論方法在情緒分類與憂鬱症風險監測任務上的效能：

**主要發現**：
1. [總結三種方法的效能排序]
2. [LoRA 微調的優勢與成本]
3. [情緒到風險映射的可行性]
4. [視覺化系統的實用性]

**研究貢獻**：
1. 提供完整的端到端實作流程
2. 展示參數高效微調在資源受限環境下的應用
3. 建立可擴展的風險監測框架
4. 深入分析模型限制與倫理考量

### 10.2 未來工作方向

#### 10.2.1 短期改進
1. **擴大評估規模**：使用完整測試集與多個資料集
2. **超參數優化**：系統性搜索最佳參數組合
3. **錯誤分析深化**：質性分析誤判案例
4. **使用者研究**：收集實際使用者回饋

#### 10.2.2 中期發展
1. **多模態融合**：
   - 整合文本、語音、影像資料
   - 考慮發文時間、頻率等元資料
   
2. **時序建模**：
   - 使用 RNN/LSTM 捕捉情緒變化趨勢
   - 預測未來風險發展軌跡
   
3. **更大規模模型**：
   - 測試 LLaMA-2-7B、Mistral-7B 等模型
   - 比較不同模型架構的效能

4. **跨語言擴展**：
   - 訓練中文等其他語言模型
   - 多語言聯合訓練

#### 10.2.3 長期願景
1. **臨床驗證研究**：
   - 與醫療機構合作收集臨床資料
   - 驗證系統預測與實際診斷的相關性
   - 長期追蹤預測準確度

2. **個人化建模**：
   - 根據個人歷史資料調整模型
   - 考慮個體差異（性別、年齡、文化背景）

3. **即時監測系統**：
   - 開發可部署的生產級系統
   - 整合警示與轉介機制
   - 建立完整的隱私保護框架

4. **可解釋性研究**：
   - 提供預測理由與支持證據
   - 幫助使用者理解與信任系統

### 10.3 最終反思

大型語言模型在情緒分析與心理健康監測領域展現出巨大潛力，但同時也帶來重要的倫理與技術挑戰。作為研究者與開發者，我們必須：

1. **謹慎設計**：充分考慮技術限制與潛在風險
2. **負責部署**：確保系統用於增進福祉而非造成傷害
3. **持續改進**：不斷驗證、更新與優化系統
4. **跨域合作**：結合 AI、心理學、醫學、倫理學等專業知識

唯有在科技進步與人文關懷之間取得平衡，我們才能真正發揮 AI 在心理健康領域的正向價值。

---

## 參考文獻 (References)

1. Saravia, E., Liu, H. C. T., Huang, Y. H., Wu, J., & Chen, Y. S. (2018). CARER: Contextualized affect representations for emotion recognition. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 3687-3697).

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.

3. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint arXiv:2305.14314*.

4. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

5. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

6. [請根據實際使用的參考資料補充]

---

## 附錄 (Appendices)

### 附錄 A：完整程式碼

請參見 GitHub repository：https://github.com/Qmo37/hw5-llm-emotion-depression

### 附錄 B：實驗環境

- **硬體**：
  - GPU: [請填寫]
  - RAM: [請填寫]
  - 儲存空間: [請填寫]

- **軟體**：
  - Python: 3.10.12
  - PyTorch: [請填寫]
  - Transformers: [請填寫]
  - PEFT: [請填寫]

### 附錄 C：超參數搜索記錄

[如有進行超參數搜索，請記錄所有嘗試的配置]

### 附錄 D：額外視覺化

[可包含其他有價值的圖表與分析]

---

**報告完成日期**：2025-11-07  
**總字數**：[請計算]  
**總頁數**：[請計算]
