import streamlit as st
import numpy as np

from preprocessing import text_preprocessing
from embeddings import load_w2v_model, avg_word2vec
from model import load_classifier, predict_message

st.set_page_config(page_title="Spam Detection", page_icon="📩")

# Load resources once
@st.cache_resource
def load_resources():
    w2v = load_w2v_model("GoogleNews-vectors-negative300.bin")
    clf = load_classifier("google_w2v_model.pkl")
    return w2v, clf

w2v_model, model = load_resources()

st.title("📩 Spam Detection using Google Word2Vec")

user_input = st.text_area("Enter message:")

if st.button("Predict"):
    if user_input.strip() != "":
        
        processed = text_preprocessing(user_input)
        vector = avg_word2vec(processed, w2v_model)

        if np.all(vector == 0):
            st.warning("No valid words found in Word2Vec vocabulary.")
        else:
            pred, prob = predict_message(model, vector)

            if pred == 1:
                st.error(f"🚨 SPAM ({prob[1]*100:.2f}% confidence)")
            else:
                st.success(f"✅ HAM ({prob[0]*100:.2f}% confidence)")
    else:
        st.warning("Please enter a message.")