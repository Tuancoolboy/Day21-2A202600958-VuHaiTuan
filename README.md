# Day 21 — Fine-tuning LLMs · Course Materials

> **AICB-P2T3 · Chương 5 — Fine-tuning & An Toàn**  
> *"Khi nào nên fine-tune — và khi nào prompt engineering đủ rồi?"*  
> Giảng viên: Nguyễn Khánh Linh · VinUniversity

---

## 📦 Cái gì có trong gói này

Đây là gói tài liệu hoàn chỉnh cho **Ngày 21** của khóa AICB Phase 2 Track 3 — bài học về fine-tuning LLMs với LoRA/QLoRA.

```
Day21_Fine-tuning/
├── 📊 Slides (LaTeX/Beamer)
│   └── day06-fine-tuning-llms-tu-full-fine-tune-en-loraqlora.tex   ← source slides
│
├── 💻 Notebooks (Google Colab)
│   └── Lab21_LoRA_Finetuning_T4.ipynb       ← cho Free Colab T4 (Qwen2.5-3B)
│
├── 📋 Tài liệu học viên
│   └── Lab21_Rubric_and_Format.md           ← rubric + submission format
│
└── 📖 README.md                              ← file này
```

---

## 🚀 Quick Start

### Cho **Giảng viên**:
1. Compile slides: `xelatex day06-fine-tuning-llms-tu-full-fine-tune-en-loraqlora.tex` (chạy 2 lần để correct page count)
2. Đọc qua `Day21_Speaker_Notes_Vietnamese.md` để chuẩn bị bài giảng
3. Test notebook trên GPU bạn có (recommend A100/L4 để demo nhanh)
4. In `Lab21_Rubric_and_Format.md` phát cho học viên

### Cho **Học viên**:
1. Đọc `Lab21_Rubric_and_Format.md` để biết deliverable + scoring
2. Chọn notebook phù hợp với GPU:
   - **Free Colab (T4 16GB)** → `Lab21_LoRA_Finetuning_T4.ipynb`
   - **Pro/A100/L4** → `Lab21_LoRA_Finetuning_BigGPU.ipynb`
3. Upload notebook lên Colab → `Runtime > Change runtime type > GPU` → `Run all`
4. Hoàn thành lab + viết REPORT.md theo template
5. Nộp zip qua LMS

### Chạy bằng Python script (code-only)

Nếu muốn chạy ngoài notebook, dùng script CLI mới:

```bash
pip install -r requirements.txt
python train_lora_qlora.py --output-dir outputs/lab21_lora_t4
```

Script sẽ tự động:
- Load dataset `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` và lấy 200 samples
- Format Alpaca, lọc output quá ngắn, dedupe, phân tích p95 token length
- Train 3 adapters `r=16`, `r=8`, `r=64` với QLoRA 4-bit
- Lưu `rank_experiment_summary.csv`, `qualitative_comparison.csv`, `loss_curve.png`, `dataset_stats.json`

Ví dụ dùng dataset JSON/JSONL tự tạo:

```bash
python train_lora_qlora.py \
  --custom-data data/my_alpaca.jsonl \
  --output-dir outputs/my_lab21 \
  --sample-size 300
```

---

## 📚 Nội dung chính của bài học

### Section 1 — Khi nào cần Fine-tune?
- **Bối cảnh 2025–2026**: frontier models đủ tốt cho 80%+ tasks
- **Pipeline ưu tiên**: Prompt Engineering → RAG → Fine-tune (theo thứ tự)
- **Quy tắc vàng**: *Fine-tune KHÔNG fix knowledge gaps — RAG cho knowledge, fine-tune cho style/format*
- **Decision tree**: gap > 15% và volume > 50k req/day → fine-tune ROI dương
- **API Fine-tuning vs Self-hosted**: trade-off giữa control, cost, và privacy

### Section 2 — LoRA: Cơ chế hoạt động
- **Ý tưởng cốt lõi**: freeze base weights, inject low-rank update $\Delta W = B \cdot A$
- **Math**: $h = W_0 x + B A x$, rank $r \ll \min(d, k)$
- **Analogy**: như dán sticky notes vào sách — không sửa sách gốc, chỉ thêm chú thích
- **Zero added latency**: merge adapter vào base weights khi deploy
- **Rank trade-off**:
  - r = 8 → 0.05% params, train nhanh
  - r = 16 → standard choice, balance tốt
  - r = 64 → near full fine-tune, tốn VRAM hơn
- **Best practice 2025**: target ALL layers (q/k/v/o + gate/up/down) thay vì chỉ q+v

### Section 3 — QLoRA: Fine-tune trên GPU nhỏ
- **Innovation**: quantize base xuống 4-bit NF4 + bf16 LoRA adapters
- **Paged AdamW**: optimizer offload sang CPU RAM khi GPU OOM
- **Double quantization**: quantize cả quantization constants
- **Cost comparison**:
  - Full FT: ~60 GB VRAM, A100 80GB, $$$$
  - LoRA fp16: ~28 GB, A100 40GB, $$
  - **QLoRA 4-bit: ~10 GB, RTX 3090 24GB, $$**
- **Multi-tenant serving**: 1 base + N adapters trên cùng GPU

### Section 4 — Training Pipeline & Infrastructure
- **Dataset prep**: Quality > Quantity (500 perfect > 10k noisy)
- **Token length analysis**: set `max_seq_length = p95` của dataset
- **VRAM math**: $V_{tot} = V_{model} + V_{optim} + V_{act} + V_{grad}$
- **Gradient checkpointing**: -60% VRAM, +20% time
- **FlashAttention**:
  - IO-aware exact attention với tiling + recomputation
  - 2-4× speedup, O(N) memory thay vì O(N²)
  - Yêu cầu Ampere+ (A100, RTX 30xx+)
- **Effective batch size**: $\text{batch} \times \text{grad\_accum}$, target 16-64
- **OOM debugging order**: max_seq_length → gradient checkpoint → rank → batch size

### Section 5 — Hands-on Demo
- Live demo: fine-tune Qwen2.5-7B trên Vietnamese domain dataset (~25 phút trên A100)
- LoRA adapter swap demo cho multi-tenant serving
- GGUF merge + llama.cpp deployment

### Lab 21 — Rank Experiment (2 giờ)
- Chuẩn bị 100-500 examples Alpaca format
- Train 3 LoRA adapters với r=8, r=16, r=64
- So sánh 4 chiều: training time, VRAM, perplexity, qualitative output
- Học hands-on về rank selection trade-off

---

## 🛠️ Stack công nghệ

| Tool | Vai trò | Version trong notebook |
|------|---------|------------------------|
| **Unsloth** | Custom CUDA kernels — 2× faster, 60% less VRAM | latest từ git |
| **TRL** | `SFTTrainer` cho supervised fine-tuning | ≥ 0.12, < 0.16 |
| **PEFT** | LoRA wrapper | latest |
| **Transformers** | Base model loading | ≥ 4.46 (cần processing_class) |
| **bitsandbytes** | 4-bit NF4 quantization | latest |
| **datasets** | Vietnamese Alpaca dataset | latest |

**Base models**:
- `unsloth/Qwen2.5-7B-bnb-4bit` (BigGPU notebook)
- `unsloth/Qwen2.5-3B-bnb-4bit` (T4 notebook)

**Reference dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (300 samples từ subset)

---

## 🎯 Kỹ năng học viên đạt được

Sau Day 21, học viên có thể:

✅ Hiểu khi nào **nên** fine-tune và khi nào **không nên**  
✅ Giải thích cơ chế LoRA + QLoRA bằng math + intuition  
✅ Train một LoRA adapter từ đầu đến cuối với production-grade tools  
✅ So sánh và chọn rank dựa trên trade-off (time / memory / quality)  
✅ Đánh giá fine-tuned model bằng perplexity + qualitative methods  
✅ Document và defend kết quả qua technical report

---

## ⚠️ Common Pitfalls — Tham khảo nhanh

Notebook đã handle các issues sau, nhưng học viên/giảng viên nên biết để debug:

| Vấn đề | Nguyên nhân | Fix |
|--------|-------------|-----|
| `KeyError: 'instruction'` | Dataset có cột `_vi` suffix | Auto-detect column names |
| `tokenizer is unexpected kwarg` | TRL <0.12 + Transformers ≥4.46 | Upgrade TRL hoặc monkey-patch |
| `evaluation_strategy is unexpected` | Transformers ≥4.46 đổi tên | Auto-detect → `eval_strategy` |
| `Dim 3 mismatch (497 vs 552)` | `packing=True` buggy với new transformers | Set `packing=False` |
| `on_train_begin must be called before on_evaluate` | NotebookProgressCallback bug | Remove callback before evaluate |
| OOM during evaluation | T4 không đủ VRAM cho eval batch | `safe_evaluate()` fallback batch=1 |

Chi tiết full troubleshooting trong `Lab21_Rubric_and_Format.md`.

---

## 📈 Bài học kế tiếp

| Day | Nội dung | Liên hệ với Day 21 |
|-----|----------|---------------------|
| **Day 22** | DPO, ORPO & Alignment | SFT dạy format → DPO/ORPO dạy alignment (helpful + safe) |
| **Day 23** | Production RAG | Khi nào dùng fine-tune vs RAG vs hybrid |
| **Day 24** | RAGAs, LLM-as-Judge, Guardrails | Đánh giá fine-tuned model rigorously |
| **Day 25** | Circuit Breakers, Reliability | Production deployment cho fine-tuned models |

---

## 📞 Hỗ trợ

- **Câu hỏi về slides / demo**: liên hệ giảng viên trên LMS
- **Câu hỏi về lab**: post lên Slack channel `#lab21-help`
- **Bug trong notebook**: báo cáo trên course repo (issues)

---

## 📜 License & Credits

- **Slides + speaker notes**: © 2026 VinUniversity AICB Program
- **Notebooks**: dựa trên Unsloth examples, modified cho khóa học AICB
- **Theme**: VinUni Beamer template (xem `theme/` directory)
- **Reference papers**:
  - LoRA — Hu et al. 2021 (https://arxiv.org/abs/2106.09685)
  - QLoRA — Dettmers et al. 2023 (https://arxiv.org/abs/2305.14314)
  - FlashAttention — Dao et al. 2022 (https://arxiv.org/abs/2205.14135)

---

> **🎓 Lưu ý cho học viên**: Lab này là một trong những hands-on quan trọng nhất của khóa. Đầu tư thời gian làm cẩn thận — kỹ năng fine-tuning LLM là một trong những skills value nhất 2025-2026 trên thị trường AI.

---

*Last updated: 2026-05*
