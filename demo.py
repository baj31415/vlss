import streamlit as st
from PIL import Image
import torch
from src.vlm.clipmodel import CLIP  
from src.vlm.clip_cppmodel import CLIPCPP
from src.vlm.blipmodel import BLIP

def load_model(model_choice):
    if model_choice == 'clip':
        model = CLIP()  
    elif model_choice == 'clip.cpp':
        model = CLIPCPP()
    elif model_choice == 'blip':
        model = BLIP()
    else:
        raise ValueError("Invalid model choice.")
    return model

def compute_embeddings(model, text_input, image):
    text_embedding = model.get_text_embedding(text_input)
    image_embedding = model.get_image_embedding(image)
    return text_embedding, image_embedding

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

st.title("Visual Language Semantic Search")

model_choice = st.selectbox("Choose a model:", ["clip", "blip", "clip.cpp"])
model = load_model(model_choice)

text_input = st.text_input("Enter text for embedding:")

uploaded_files = st.file_uploader("Upload images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if text_input and uploaded_files:
    text_embedding = model.get_text_embedding(text_input)
    image_embeddings = []
    images = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images.append(image)
        image_embedding = model.get_image_embedding(image)
        image_embeddings.append(image_embedding)

    similarities = [cosine_similarity(text_embedding, img_emb) for img_emb in image_embeddings]
    

    sorted_images = [img for _, img in sorted(zip(similarities, images), key=lambda pair: pair[0], reverse=True)]
    
    st.write("## Closest images to the text input")
    for img in sorted_images:
        st.image(img)