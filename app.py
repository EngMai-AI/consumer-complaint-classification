import re, nltk, torch, gradio as gr
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:  # only needed on Hugging Face ZeroGPU Spaces
    import spaces

    @spaces.GPU
    def _gpu_noop():
        return 0
except ImportError:
    pass

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOP = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

MODEL_ID = "mai-data-ai/complaint-distilbert"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

def clean_text(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(lemm.lemmatize(w) for w in text.split() if w not in STOP)

def predict(text):
    if not text or not text.strip():
        return {}
    inputs = tok(clean_text(text), truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, label="Complaint narrative"),
    outputs=gr.Label(num_top_classes=5, label="Predicted category & confidence"),
    title="Consumer Complaint Classifier",
    description="DistilBERT fine-tuned on consumer complaints (5 categories).",
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch()
