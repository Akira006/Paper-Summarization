# Research Paper Summarization Using Long-Document Transformer Models

Eksperimen perbandingan model-model summarization untuk merangkum artikel ilmiah secara otomatis, dengan fokus pada kemampuan model dalam menangani dokumen panjang.

---

## Overview

Artikel ilmiah punya karakteristik yang bikin summarization jadi susah — panjang, padat informasi, dan struktur yang kompleks. Project ini mencoba menjawab pertanyaan: **apakah model yang dirancang khusus untuk dokumen panjang (seperti LED) memang lebih efektif dibanding model standar?**

Untuk itu, lima model dibandingkan secara langsung menggunakan dataset PubMed dan metrik evaluasi ROUGE.

---

## Models

| Model | Tipe | Max Input | Keterangan |
|-------|------|-----------|------------|
| TextRank | Extractive Baseline | — | Tanpa training, graph-based ranking |
| BART | Abstractive Baseline | 1024 token | Denoising autoencoder |
| PEGASUS | Abstractive Baseline | 1024 token | Gap-sentence generation pretraining |
| LongT5 | Long-Document | 4096 token | Transient-global attention |
| LED | Long-Document (Main) | 8192 token | Sparse attention, model utama |

---

## Dataset

**ccdv/pubmed-summarization** — subset PubMed dari Scientific Papers Dataset (Cohan et al., 2018)

- Train: ~119,924 artikel
- Validation: 6,633 artikel  
- Test: 6,658 artikel
- Input: full article text → Target: abstract

---

## Pipeline

Setiap model mengikuti pipeline yang sama:

```
Dataset Preparation → Text Pre-processing → Training → Inference → Evaluation
```

1. **Dataset Preparation** — load dataset, mapping kolom, subset untuk eksperimen
2. **Pre-processing** — cleaning, normalization, tokenization per model
3. **Training** — fine-tuning dengan Seq2SeqTrainer (kecuali TextRank)
4. **Inference** — generate ringkasan dari test set
5. **Evaluation** — ROUGE-1, ROUGE-2, ROUGE-L + evaluasi kualitatif

---

## Evaluation

**Kuantitatif:** ROUGE-1, ROUGE-2, ROUGE-L (F1 score)

**Kualitatif (manual):**
- Coverage — apakah ringkasan mencakup tujuan, metode, hasil, dan kesimpulan
- Coherence — apakah ringkasan mudah dipahami dan alurnya logis
- Factual Consistency — apakah isi ringkasan sesuai artikel sumber
- Hallucination — apakah ada informasi yang tidak ada di artikel asli

---

## Repository Structure

```
├── notebooks/
│   ├── nlp-bart-summarization.ipynb
│   ├── nlp-pegasus-summarization.ipynb
│   ├── nlp-textrank-summarization.ipynb
│   ├── nlp-longt5-summarization.ipynb
│   ├── nlp-led-summarization.ipynb
│   └── nlp-comparative-analysis.ipynb
├── results/
│   ├── bart_rouge_results.csv
│   ├── pegasus_rouge_results.csv
│   ├── textrank_rouge_results.csv
│   ├── longt5_rouge_results.csv
│   └── led_rouge_results.csv
├── proposal/
│   └── Proposal_NLP_Kelompok_12.pdf
├── .gitignore
└── README.md
```

---

## Environment

- Python 3.10+
- PyTorch + CUDA (T4 GPU)
- `transformers`, `datasets`, `rouge-score`, `sentencepiece`, `accelerate`
- Training environment: Kaggle / Google Colab

---

## References

- Beltagy et al. (2020) — Longformer: The Long-Document Transformer
- Lewis et al. (2020) — BART
- Zhang et al. (2020) — PEGASUS
- Guo et al. (2022) — LongT5
- Mihalcea & Tarau (2004) — TextRank
- Lin (2004) — ROUGE
