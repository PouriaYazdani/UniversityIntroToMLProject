# Multi-Label Emotion Classification on Sentences

A fine-tuned DistilBERT model for multi-label emotion classification on short English sentences.  
The task involves predicting 10 possible emotions per sentence, where multiple emotions can co-occur.

## Project Overview

The objective is to classify sentences into one or more of the following 10 emotion categories:

- praise
- amusement
- anger
- disapproval
- confusion
- interest
- sadness
- fear
- joy
- love

Due to limited computational resources, DistilBERT-base-uncased was selected as the backbone (66M parameters, ~97% of BERT-base performance) instead of heavier alternatives such as BERT-base, RoBERTa-base, or larger models. A lightweight classification head (pre-classifier → Tanh → Dropout → Linear) was added on top of the [CLS] representation.

## Dataset & Exploratory Data Analysis

The provided dataset exhibits the following characteristics:

- Severe class imbalance: `confusion` and `praise` dominate both training and validation splits.
- Multi-label nature: approximately 65% of samples have exactly one label, ~30% have two labels, and a small portion have three or more.
- Sentence length distribution: nearly all sentences are shorter than 30 tokens; a maximum length of 128 comfortably covers the entire dataset with negligible truncation.
- Presence of emojis: the 20 most frequent emojis were inspected; they were left untouched as they primarily correlate with `joy` and `amusement`.
- Train dataset size: 147900, Validation dataset size: 26100, Test dataset size: 26664 (kept class imbalance across splits, details in `emotion_mlcls_final.ipynb`.)

<img width="1617" height="593" alt="image" src="https://github.com/user-attachments/assets/72d8e158-e44b-42d4-b2f7-710ba7bb6ffc" />
<p align="center">
  <img width="500" height="371" alt="image" src="https://github.com/user-attachments/assets/52909888-0ab0-4e4f-a015-b1be029b2609" />
</p>

## Model Architecture

```text
DistilBertModel (distilbert-base-uncased)
   ↓
[CLS] token embedding (768-dim)
   ↓
Linear(768 → 768) + Tanh
   ↓
Dropout(p=0.1)
   ↓
Linear(768 → 10)
```

Training uses BCEWithLogitsLoss with label smoothing. AdamW optimizer and linear learning-rate scheduling with warmup are applied.

## Hyperparameter Search

Two systematic hyperparameter searches were conducted using Optuna:

1. Standard Search  (model and training hyperparamters)
   Optimized: learning rate, batch size, weight decay, hidden/attention dropout probabilities, label smoothing factor, warmup ratio, and number of epochs. Early stopping was based on validation macro-F1.

   Trial outputs and logs are [here (best model at trial 1)](https://drive.google.com/drive/folders/1-2mV7lyEfPDMSY-5h11BcSa2k_xZOufn)).

3. Search with Per-Class BCE Weighting  
   In addition to the above, 10 positive class weights (one per emotion) were jointly optimized to directly address class imbalance. These weights are multiplied by the standard BCE loss term for positive samples, giving the model stronger gradients for underrepresented emotions.

   Trial outputs and logs are [here](https://drive.google.com/drive/folders/12gGe4MZ7qtfAibAcNlj5PRdpVCzGjW_c).



However the first approach yielded the best validation loss and was selected for final training.

Below is the Training-validation curve of the best trial (trained on 147900 Train Dataset)

<p align="center">
  <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/34657e33-370f-4be7-862a-d2fe5edf98a0" />
</p>



Final model with best set of hyperparameters trained on full data set (train+validation+test) is [here](https://drive.google.com/drive/folders/1_9OXh7ciySxWV43NMJgqoRcWfvaonTB0).

## Results on Held-Out Test Set (FinalTestEmo_1000)

| Metric              | Score    |
|---------------------|----------|
| Macro-F1            | 0.4906   |
| Micro-F1            | 0.6123   |
| ROC-AUC (macro)     | 0.7410   |
| Hamming Score       | 0.8736   |

Per-class F1 scores and a detailed confusion matrix analysis are available in the notebook. The model struggles most with `confusion` despite it being the most frequent label, indicating possible label noise.

## Streamlit Demo

A lightweight inference application is provided:
<p align="center">
  <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/6ec86e56-94cd-41e8-abf4-9c760246e4bf" />
</p>

To run the demo locally:

```bash
pip install streamlit torch transformers
streamlit run app.py
```

(Update `MODEL_PATH` in `app.py` to point to the saved `best_model.pt` checkpoint.)

## Repository Structure

```
├── data/                              # Dataset files and saved plots
├── streamlit_ui/
│   ├── app.py                         # Streamlit inference app
│   ├── model.py                       # Model architecture definition
│   └── streamlit_demo.png            # Screenshot of the web demo
├── emotion_mlcls_final.ipynb          # Complete notebook (EDA → training → evaluation → error analysis)
└── README.md
```
---
Project completed as the final project for the Natural Language Processing course (2023–2024).
```
