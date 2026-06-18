# Research Paper Summarization: A Comparative Study of Five Summarization Models

Eksperimen perbandingan model-model summarization untuk merangkum artikel ilmiah secara otomatis, dengan fokus pada kemampuan model dalam menangani dokumen panjang.

---

## Overview

Artikel ilmiah punya karakteristik yang bikin summarization jadi susah — panjang, padat informasi, dan struktur yang kompleks. Project ini membandingkan lima pendekatan summarization yang berbeda untuk melihat bagaimana performa masing-masing model pada tugas merangkum research paper.

Tujuannya bukan untuk memposisikan satu model sebagai baseline utama, melainkan **mengeksplorasi dan membandingkan semua model secara setara** — mulai dari extractive sederhana (TextRank), abstractive standar (BART, PEGASUS), hingga model long-document (LongT5, LED) — menggunakan dataset PubMed dan metrik evaluasi ROUGE serta evaluasi kualitatif manual.

---

## Models

| Model | Tipe | Max Input | Keterangan |
|-------|------|-----------|------------|
| TextRank | Extractive | — | Tanpa training, graph-based ranking via `sumy` |
| BART | Abstractive | 1024 token | `facebook/bart-large-cnn`, denoising autoencoder |
| PEGASUS | Abstractive | 512 token | `google/pegasus-xsum`, gap-sentence generation |
| LongT5 | Long-Document | 4096 token | `google/long-t5-tglobal-base`, transient-global attention |
| LED | Long-Document | 8192 token | `allenai/led-base-16384`, sparse attention |

> Catatan: panjang input maksimal tiap model disesuaikan dengan karakteristik arsitekturnya sekaligus mempertimbangkan keterbatasan VRAM (GPU T4 15GB).

---

## Dataset

**ccdv/pubmed-summarization** — subset PubMed dari Scientific Papers Dataset (Cohan et al., 2018), diakses sebagai alternatif kompatibel dari `armanc/scientific_papers` yang sudah deprecated.

Dataset penuh:
- Train: ~119,924 artikel
- Validation: 6,633 artikel
- Test: 6,658 artikel

Input: full article text (`article`) ke Target: abstract (`abstract`), dipetakan menjadi `input_text` dan `target_text`.

**Sampling untuk eksperimen:** karena keterbatasan komputasi (GPU T4 15GB), digunakan subset data yang konsisten di semua model — **6.658 sampel untuk training**, dan **500 sampel masing-masing untuk validation dan test**. Pendekatan ini menjaga perbandingan tetap fair antar model sambil tetap feasible secara waktu dan resource.

---

## Pipeline

Setiap model mengikuti pipeline yang sama:

```
Dataset Preparation -> Text Pre-processing -> Training -> Inference -> Evaluation -> Comparative Analysis
```

1. **Dataset Preparation** — load dataset, mapping kolom, sampling subset untuk eksperimen
2. **Pre-processing** — cleaning (rapikan whitespace, hapus karakter kontrol), tokenization per model. Tidak menggunakan stemming/lemmatization/stopword removal karena Transformer membutuhkan struktur bahasa yang utuh
3. **Training** — fine-tuning dengan `Seq2SeqTrainer` (kecuali TextRank yang tidak memerlukan training)
4. **Inference** — generate ringkasan dari test set menggunakan beam search
5. **Evaluation** — ROUGE-1, ROUGE-2, ROUGE-L + evaluasi kualitatif manual
6. **Comparative Analysis** — gabungkan hasil semua model untuk perbandingan

---

## Evaluation

**Kuantitatif:** ROUGE-1, ROUGE-2, ROUGE-L (F1 score)

**Kualitatif (manual):**
- Coverage — apakah ringkasan mencakup tujuan, metode, hasil, dan kesimpulan
- Coherence — apakah ringkasan mudah dipahami dan alurnya logis
- Factual Consistency — apakah isi ringkasan sesuai artikel sumber
- Hallucination — apakah ada informasi yang tidak didukung artikel asli

Setiap notebook juga menghasilkan visualisasi: ROUGE bar chart, distribusi score per sampel (histogram & boxplot), analisis panjang input vs score, dan training curves.

---

## Repository Structure

```
├── Notebook/
│   ├── nlp-bart-summarization.ipynb
│   ├── nlp-pegasus-summarization.ipynb
│   ├── nlp-textrank-summarization.ipynb
│   ├── nlp-longt5-summarization.ipynb
│   ├── nlp-led-summarization.ipynb
│   └── All-Model comparison/
│       └── Master-Comparison.ipynb
├── Result/
│   ├── BART/
│   ├── PEGASUS/
│   ├── LongT5/
│   ├── LED/
│   ├── Textrank/
│   └── All Models/
├── proposal/
│   └── Proposal_NLP_Kelompok_12.pdf
├── .gitignore
└── README.md
```

Setiap folder di `Result/` berisi `*_rouge_results.csv`, `*_predictions.json`, dan file visualisasi `.png`. Folder `All Models/` berisi hasil perbandingan agregat.

> Model weights (`*.safetensors`) tidak disertakan di repo karena ukurannya besar (>1GB). Weights disimpan terpisah di Google Drive.

---

## Web Deployment

Model BART hasil fine-tuning dideploy sebagai aplikasi web interaktif menggunakan **Streamlit** untuk demo summarization secara langsung.

---

## Environment

- Python 3.10+
- PyTorch + CUDA (GPU T4)
- `transformers`, `datasets`, `rouge-score`, `sentencepiece`, `accelerate`, `sumy`, `matplotlib`
- Training environment: Google Colab (via VS Code tunnel)

---

## Catatan Teknis

Beberapa kendala dan solusi yang ditemui selama pengembangan:

- `armanc/scientific_papers` tidak lagi didukung di `datasets >= 4.0` (loading script deprecated) -> diganti ke `ccdv/pubmed-summarization`
- LongT5 dan LED tidak kompatibel dengan `fp16` -> menggunakan `bf16` + `gradient_checkpointing` + optimizer `adafactor` untuk efisiensi VRAM
- LED membutuhkan `global_attention_mask` baik saat training maupun inference
- Inference menggunakan `.generate()` langsung (bukan `pipeline()`) untuk menghindari breaking changes pada nama task

---

## References

- Beltagy et al. (2020) — Longformer: The Long-Document Transformer
- Lewis et al. (2020) — BART
- Zhang et al. (2020) — PEGASUS
- Guo et al. (2022) — LongT5
- Mihalcea & Tarau (2004) — TextRank
- Lin (2004) — ROUGE
- Cohan et al. (2018) — A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents
- Fabbri et al. (2021) — SummEval
- Kryscinski et al. (2020) — Evaluating the Factual Consistency of Abstractive Text Summarization
