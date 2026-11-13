# Consumer Complaint Classification

End-to-end NLP pipeline that classifies customer complaints into 5 product categories, comparing **SimpleRNN, LSTM, GRU** (built from scratch in Keras) against a **fine-tuned DistilBERT** (Hugging Face).

- 🤗 Model: [mai-data-ai/complaint-distilbert](https://huggingface.co/mai-data-ai/complaint-distilbert)
- 🚀 Live demo: [mai-data-ai/complaint-classifier](https://huggingface.co/spaces/mai-data-ai/complaint-classifier)

## Objective
Automatically predict the correct complaint category from a customer's written complaint, to improve routing efficiency and reduce manual effort and cost.

Categories: `credit_card`, `credit_reporting`, `debt_collection`, `mortgages_and_loans`, `retail_banking`.

## Dataset
[Kaggle Consumer Complaints Dataset for NLP](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp): 162,421 rows → **124,300** after removing empty narratives, exact duplicates (37,735) and texts with conflicting labels (172).

Class distribution is imbalanced (`credit_reporting` ≈ 45%), handled with class-weighted loss. Split: stratified 80/10/10 (99,440 / 12,430 / 12,430), seed 42.

## Pipeline
1. Load and explore data, analyze class distribution and text length
2. Preprocessing: lowercase, remove punctuation/numbers/special characters, remove stop words, lemmatization
3. Tokenization (Keras `Tokenizer`, vocab 30,000) and padding (max length 250)
4. Embedding layer (128-d) → SimpleRNN / LSTM / GRU → Dense classifier
5. Fine-tune `distilbert-base-uncased` (lr 2e-5, 3 epochs, batch 32, max length 256, class-weighted loss)
6. Evaluate: accuracy, precision, recall, F1, confusion matrix
7. Compare models and classify new complaints

## Results (test set, 12,430 complaints)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Params |
|---|---|---|---|---|---|
| SimpleRNN | 0.7896 | 0.7659 | 0.7631 | 0.7598 | 3.88M |
| LSTM | 0.8263 | 0.7987 | 0.8286 | 0.8120 | 3.98M |
| GRU | 0.8307 | 0.8023 | 0.8462 | 0.8212 | 3.95M |
| **DistilBERT** | **0.8612** | **0.8499** | **0.8486** | **0.8491** | 66.96M |

**Best model: DistilBERT.** It gains about +3 points of accuracy and +2.8 points of macro-F1 over the best from-scratch model (GRU). Among the from-scratch models, GRU is best and also trained fastest (≈0.9 min vs 1.5 for LSTM and 2.0 for SimpleRNN).

Most errors (1,725 in total) come from overlapping categories: `credit_reporting` → `debt_collection` (325) and `debt_collection` → `credit_reporting` (295).

> Inference times in `results/` were measured on different hardware for the Keras models and DistilBERT, so they are not directly comparable.

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification",
               model="mai-data-ai/complaint-distilbert", top_k=None)
print(clf("debt collector keep calling loan already paid", truncation=True, max_length=256))
```

Input text should be cleaned like the training data (see `clean_text` in the notebooks).

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

## Repository structure
`notebooks/` experiments and training · `app.py` Gradio app · `results/` metrics and figures

## Limitations
- Overlapping categories (mainly credit reporting vs debt collection)
- English text from a single source only
- Meant to assist routing, not to replace human review

## License
MIT
