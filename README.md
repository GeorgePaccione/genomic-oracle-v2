# Genomic Oracle v2: Cascaded Hybrid Machine Learning Pipeline

Genomic Oracle v2 is an automated, multi-stage intelligent agent pipeline designed to ingest raw, unstructured nucleotide sequences, accurately classify functional biological domains (promoters and coding regions), and dynamically enrich the results with live external database metadata via REST APIs.

The system is deployed as an interactive web application on Hugging Face Spaces leveraging ZeroGPU for dynamic hardware allocation.

## Key Features
* **Hybrid Architecture:** Blends the semantic understanding of DNA language models (DNABERT-2) with the speed and structural efficiency of classic machine learning (Logistic Regression, LightGBM).
* **Compute-Optimized Cascade:** Implements an upstream probability-thresholding filter that routes ambiguous sequences to heavy deep learning models while resolving standard sequences instantly on CPU scales.
* **Live API Data Enrichment:** Automated asynchronous hooks fetch functional annotations from the NCBI and Ensembl databases, rendering real-time structured data insights.

## Architecture Overview
The inference pipeline consists of a 4-stage neuro-symbolic cascade:
1. **Feature Extraction:** Raw DNA text strings are embedded using a pre-trained `DNABERT-2` language model.
2. **The Gatekeeper (Stage 1):** A lightweight `scikit-learn` Logistic Regression model screens high-dimensional embeddings to immediately discard genomic noise.
3. **The Structural Mapper (Stage 2):** A `LightGBM` gradient-boosted decision tree parses non-linear sequence boundaries and patterns.
4. **Deep Contextual Refiner (Stages 3 & 4):** Complex sequences are routed to a fine-tuned `DNABERT-2` promoter model and a custom ALiBi BERT phenotype classifier designed for extended context window processing.

## Tech Stack & Dependencies
* **Core Language:** Python 3.10+
* **Machine Learning:** PyTorch, LightGBM, Scikit-Learn, Transformers (Hugging Face)
* **Data Pipelines & UI:** Pandas, NumPy, Gradio, REST APIs (Requests)
* **Deployment:** Hugging Face Spaces (ZeroGPU backend)

## Installation & Local Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/genomic-oracle-v2.git](https://github.com/YOUR_USERNAME/genomic-oracle-v2.git)
   cd genomic-oracle-v2
