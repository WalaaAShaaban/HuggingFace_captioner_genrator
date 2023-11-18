import streamlit as st
from transformers import pipeline
from PIL import Image

captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")
generator = pipeline('text-generation', model = 'gpt2')

uploaded_image = st.file_uploader("Choose a CSV file", type=['jpg', 'png', 'jpeg'])

if uploaded_image:
    image = Image.open(uploaded_image)
    caption = captioner(image)
    st.image(image)
    st.write(caption[0]['generated_text'])
    story = generator(caption[0]['generated_text'], max_length = 50, num_return_sequences=3)
    st.write(story)
