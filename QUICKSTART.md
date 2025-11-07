# Quick Start Guide - HW5

Get started with HW5 in 5 minutes!

## Step 1: Access the Notebook

### Option A: Google Colab (Recommended)
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `HW5_LLM_Emotion_Depression.ipynb`
4. **Important**: Enable GPU
   - Click **Runtime → Change runtime type**
   - Set **Hardware accelerator** to **GPU**
   - Click **Save**

### Option B: Clone from GitHub
```bash
git clone https://github.com/Qmo37/hw5-llm-emotion-depression.git
cd hw5-llm-emotion-depression
```

Then upload to Colab or use Jupyter locally.

---

## Step 2: Run the Notebook

**In Colab**:
1. Click **Runtime → Run all** (or press Ctrl+F9)
2. Wait for packages to install (~2-3 minutes)
3. The notebook will run automatically (~1-2 hours total)

**Cell by Cell** (for learning):
- Press **Shift+Enter** to run each cell
- Read the markdown explanations between code cells
- Watch the outputs and understand each step

---

## Step 3: What You'll Get

After running all cells, you'll have:

### Generated Files:
- `lora_emotion_final/` - Your trained LoRA model
- `method_comparison.png` - Performance comparison chart
- `confusion_matrices.png` - 6 confusion matrices
- `risk_trend.png` - Risk probability trend
- `risk_heatmap.png` - Risk concentration heatmap
- `risk_comprehensive.png` - Multi-level risk analysis
- `HW5_Summary_Report.txt` - Results summary

### Results in Notebook:
- Dataset statistics and distributions
- Zero-shot performance metrics
- Few-shot performance metrics  
- LoRA training progress
- Comprehensive comparison table
- All visualizations embedded

---

## Step 4: Write Your Report

Use the template `Technical_Report_Template.md`:

1. **Fill in Basic Info**:
   - Your name, student ID, date

2. **Complete Sections with Results**:
   - Abstract: Summarize your findings
   - Dataset Description: Copy statistics from notebook
   - Results: Insert performance tables
   - Visualizations: Add generated images
   - Discussion: Analyze what you learned

3. **Export to PDF/Word**:
   - Copy markdown content to Word/Google Docs
   - Add images
   - Format properly
   - Export as PDF

---

## Step 5: Submit

Submit two items:

1. **GitHub Link**:
   ```
   https://github.com/Qmo37/hw5-llm-emotion-depression
   ```

2. **Technical Report**:
   - Your completed report (PDF or Word)

---

## Common Issues & Solutions

### Issue 1: Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size: Change `per_device_train_batch_size=4` to `2`
- Use smaller model: Try `TinyLlama-1.1B` instead of larger models
- Restart runtime: **Runtime → Factory reset runtime**

### Issue 2: Slow Execution
**Problem**: Cells taking too long

**Solution**:
- Reduce evaluation samples: Change `num_samples=500` to `200`
- Skip some evaluations temporarily
- Make sure GPU is enabled (check with `!nvidia-smi`)

### Issue 3: Package Installation Fails
**Error**: `ERROR: Could not find a version that satisfies the requirement...`

**Solution**:
```python
!pip install --upgrade pip
!pip install -q transformers datasets accelerate peft bitsandbytes
```

### Issue 4: Model Download Slow
**Problem**: HuggingFace downloads are slow

**Solution**:
- Be patient (first time ~5-10 minutes)
- Models are cached after first download
- Check internet connection

### Issue 5: Can't Find Generated Files
**Problem**: Where are my PNG files?

**Solution**:
- In Colab: Look in left sidebar **Files** section
- Download: Right-click file → Download
- Or run: `!zip -r outputs.zip *.png lora_emotion_final/`

---

## Understanding the Results

### Performance Metrics

**F1 Score**:
- Range: 0.0 to 1.0 (higher is better)
- Good: > 0.7
- Excellent: > 0.8

**Method Comparison**:
- Expected order: LoRA > Few-shot > Zero-shot
- LoRA should be 10-20% better than Zero-shot

### Confusion Matrix

- **Diagonal** (top-left to bottom-right): Correct predictions
- **Off-diagonal**: Mistakes
- Look for patterns: Which emotions get confused?

### Risk Monitoring

- **High Risk Probability**: Should be low for healthy populations
- **Threshold (0.3)**: Samples above may need attention
- **Trends**: Look for sudden increases or sustained high levels

---

## Tips for Better Results

### 1. Experiment with Prompts
Try different prompt templates in the Zero-shot and Few-shot sections:
```python
# Current
"Classify the following text into one of these emotions..."

# Alternative
"As an expert emotion analyst, determine which emotion best describes..."
```

### 2. Adjust LoRA Parameters
```python
# More capacity (slower, better)
r=32, lora_alpha=64

# Less capacity (faster, may be sufficient)
r=8, lora_alpha=16
```

### 3. Increase Training Epochs
```python
# Current
num_train_epochs=3

# For better results
num_train_epochs=5
```

### 4. Use Full Test Set
```python
# Current (for speed)
num_samples=200

# For accurate evaluation
num_samples=len(dataset['test'])  # Full 2000
```

---

## Time Estimates

| Task | Duration | Can Skip? |
|------|----------|-----------|
| Package installation | 2-3 min | No |
| Dataset loading | 1 min | No |
| Zero-shot (200 samples) | 10-15 min | Yes (reduce samples) |
| Few-shot (200 samples) | 15-20 min | Yes (reduce samples) |
| LoRA training (3 epochs) | 30-45 min | No |
| LoRA evaluation | 15-20 min | Yes (reduce samples) |
| Visualizations | 5-10 min | No |
| **Total** | **~1.5-2 hours** | |

---

## Key Sections to Focus On

### For Understanding:
1. **Section 2**: Dataset exploration - See data distribution
2. **Section 3**: Risk mapping - Understand the concept
3. **Section 8**: Results comparison - Main findings

### For Coding Skills:
1. **Section 5**: Zero-shot prompting techniques
2. **Section 6**: Few-shot learning implementation
3. **Section 7**: LoRA configuration and training

### For Report Writing:
1. **Section 8**: All evaluation metrics
2. **Section 9**: Visualization interpretation
3. **Section 10**: Summary and conclusions

---

## Customization Ideas

### Easy Modifications:
1. Change model to GPT-2 or other small models
2. Adjust max_length for longer/shorter texts
3. Modify risk mapping rules
4. Add more demonstration examples in Few-shot

### Advanced Modifications:
1. Implement additional metrics (precision, recall)
2. Add cross-validation
3. Try different LoRA target modules
4. Implement ensemble methods

---

## Getting Help

1. **Read the Docs**:
   - `README.md` - Complete documentation
   - `Technical_Report_Template.md` - Detailed guidance
   - `IMPLEMENTATION_SUMMARY.md` - Overview

2. **Check Outputs**:
   - Look at error messages carefully
   - Check `print()` statements in code
   - Verify GPU is being used

3. **Debug Step by Step**:
   - Run cells one at a time
   - Check intermediate outputs
   - Use `print()` to inspect variables

4. **Ask for Help**:
   - Course instructor/TA
   - GitHub Issues
   - Study group

---

## Success Checklist

Before submission, verify:

- [ ] Notebook runs without errors
- [ ] All 6 PNG files generated
- [ ] LoRA model saved in `lora_emotion_final/`
- [ ] Results table shows all three methods
- [ ] Technical report completed
- [ ] Report includes all required sections
- [ ] Figures embedded in report
- [ ] GitHub repository accessible
- [ ] Repository has README.md
- [ ] All code committed and pushed

---

## Final Tips

1. **Start Early**: Don't wait until deadline
2. **Save Frequently**: Download outputs as you go
3. **Read Carefully**: Understand what each cell does
4. **Experiment**: Try different settings
5. **Document**: Take notes for your report
6. **Ask Questions**: Don't hesitate to seek help

---

## Ready to Start?

1. Open Colab: https://colab.research.google.com/
2. Upload `HW5_LLM_Emotion_Depression.ipynb`
3. Enable GPU
4. Click **Runtime → Run all**
5. Take a coffee break ☕
6. Come back to amazing results! 🎉

---

**Good luck with your assignment!** 🚀

If you found this helpful, give the repository a ⭐ on GitHub!

Repository: https://github.com/Qmo37/hw5-llm-emotion-depression
