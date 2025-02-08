import gradio as gr
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_news(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=500, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)[0][0]
    return "Real" if prediction > 0.5 else "Fake"

interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(label="Enter News Text"),
    outputs=gr.Label(label="Prediction"),
    title="Fake News Detector",
    description="Enter a piece of news to check if it is real or fake."
)

interface.launch()
