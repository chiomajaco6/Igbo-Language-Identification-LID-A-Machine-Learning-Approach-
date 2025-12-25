import joblib
import re
import gradio as gr

# Load artifacts
model = joblib.load("lid_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ſ\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_language(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    if pred == "igbo":
        return "Igbo"
    else:
        return "Non-Igbo"

interface = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here"),
    outputs="text",
    title="Igbo Language Identification",
    description="Detects whether a text is Igbo or non-Igbo (Yoruba/Hausa)."
)

interface.launch()
