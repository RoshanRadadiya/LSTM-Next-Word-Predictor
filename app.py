import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('nextword_model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}
max_len = 167

def generate_text(seed_text, num_words=10):
    text = seed_text
    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]
        padded = pad_sequences([seq], maxlen=max_len, padding='pre')
        predicted = model.predict(padded, verbose=0)
        pos = np.argmax(predicted)
        next_words = reverse_index.get(pos, " ")
        text += ' ' + next_words
    return text

# ---- Simple CSS for button and title ----
st.markdown("""
    <style>
    /* Green button */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
    }
    /* Title on one line */
    h2 {
        display: inline-block;
        white-space: nowrap;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title (stays on one line) ----
st.markdown("<h2>📖 Next Word Prediction with LSTM</h2>", unsafe_allow_html=True)

# ---- Input widgets ----
seed = st.text_input("Enter a seed text:", "People are")
num_words = st.slider("Number of words to generate:", 1, 20, 10)

# ---- Generate with spinner and readable output box ----
if st.button("Generate"):
    with st.spinner("Generating..."):
        result = generate_text(seed, num_words)
    
    # Output box: light background + black text + monospace
    st.markdown(
        f'<div style="background-color:#f4f4f4; padding:12px; border-radius:8px; '
        f'font-family: monospace; font-size:1rem; color:#000000;">{result}</div>',
        unsafe_allow_html=True
    )