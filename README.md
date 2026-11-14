Consumer Complaint Classification

End-to-end NLP pipeline that classifies customer complaints into 5 product categories. It compares SimpleRNN, LSTM and GRU (built from scratch in Keras) against a fine-tuned DistilBERT (Hugging Face).

🤗 Model: mai-data-ai/complaint-distilbert
🚀 Live demo (Gradio): mai-data-ai/complaint-classifier
Objective

Automatically predict the correct complaint category from a customer's written complaint, to improve routing efficiency and reduce manual effort and cost.

Categories: credit_card, credit_reporting, debt_collection, mortgages_and_loans, retail_banking.

Dataset

Consumer Complaints Dataset for NLP (Kaggle): 162,421 rows, reduced to 124,300 after removing empty narratives, exact duplicates and texts with conflicting labels.

The text in the dataset is already cleaned (lowercase, no punctuation or numbers, no stop words, lemmatized). The notebook includes a clean_text function that reproduces the same cleaning for new input, and checks that it matches the dataset.
The classes are imbalanced (credit_reporting is about 45%), so training uses class weights computed from the training set only.
Split: stratified 80/10/10 (99,440 train / 12,430 validation / 12,430 test), seed 42.
Pipeline
Load and explore the data; analyze class distribution and text length
Preprocessing (lowercase, remove punctuation/numbers/special characters, remove stop words, lemmatization)
Tokenization (Keras Tokenizer, vocabulary 30,000) and padding (max length 250)
Embedding layer (128-d) → SimpleRNN / LSTM / GRU (128 units) → Dense classifier
Fine-tune distilbert-base-uncased (lr 2e-5, 3 epochs, batch size 32, max length 256, class-weighted loss, best checkpoint by validation macro-F1)
Evaluate with accuracy, precision, recall, F1 and confusion matrix
Compare the four models and classify new complaints
Deploy with Gradio (Hugging Face Space)
Results

Test set, 12,430 complaints, as produced by the notebook:

Model	Accuracy	Macro-P	Macro-R	Macro-F1	Params	Train time (min)
SimpleRNN	0.7973	0.7689	0.7855	0.7747	3.88M	1.6
LSTM	0.8191	0.7938	0.8231	0.8063	3.98M	1.9
GRU	0.8250	0.8001	0.8453	0.8184	3.95M	0.9
DistilBERT	0.8550	0.8313	0.8604	0.8447	66.96M	30.8

Best model: DistilBERT. It improves on the best from-scratch model (GRU) by about 3 points of accuracy and 2.6 points of macro-F1. Among the from-scratch models, GRU is the best and also the fastest to train.

The DistilBERT checkpoint published on the Hugging Face Hub is a separate training run. Measured on the same test set, it scores accuracy 0.8612 and macro-F1 0.8491, so the numbers in its model card differ slightly from the table above.

Most errors come from overlapping categories. In the notebook run, credit_reporting → debt_collection is the largest confusion (399 of 1,802 errors), followed by credit_reporting → mortgages_and_loans (250) and debt_collection → credit_reporting (180).

Usage
python
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

nltk.download("stopwords"); nltk.download("wordnet")
STOP = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(lemm.lemmatize(w) for w in text.split() if w not in STOP)

clf = pipeline("text-classification",
               model="mai-data-ai/complaint-distilbert", top_k=None)

complaint = "A debt collector keeps calling me about a loan I already paid off."
print(clf(clean_text(complaint), truncation=True, max_length=256))

Input text should be cleaned the same way as the training data.

Run the notebook
bash
pip install -r requirements.txt
jupyter notebook Consumer_Complaint_Classification.ipynb

DistilBERT fine-tuning was done on a GPU in Google Colab. Training on CPU will be very slow.

Run the Gradio app locally
bash
pip install -r requirements.txt
python app.py

The app downloads the model from the Hugging Face Hub on first run and returns the predicted category with its confidence score.

Repository structure
Consumer_Complaint_Classification.ipynb   data exploration, training, evaluation, comparison
app.py                                    Gradio app (same code as the live demo)
requirements.txt                          dependencies
README.md
LICENSE
Limitations
Categories overlap, mainly credit reporting vs. debt collection
English text from a single source (US consumer complaints); performance on other domains or languages is not guaranteed
Only 5 coarse product categories; it does not predict issue type or sub-product
Meant to assist routing, not to replace human review
License

MIT
