import requests
headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}  # <-- the key is here!
r = requests.get("https://www.dropbox.com/scl/fi/r30jjkq33qftgglpcehvb/use_model.h5?rlkey=7nm0qqtno4hkvidjt03jb8cgs&dl=0", stream=True, headers=headers)
with open("use_model.h5", 'wb') as f:
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)
import streamlit as st
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_hub as hub
from tensorflow.keras import layers
import cv2
import os
import csv
from pathlib import Path
import pytesseract
import re
import pandas as pd
import numpy as np

def remove_newlines_and_extra_spaces_and_save(text, filename="cleaned_text.txt"):
    # Remove newlines
    text_without_newlines = text.replace("\n", " ")
    # Remove extra spaces between words
    text_without_extra_spaces = re.sub(r'\s+', ' ', text_without_newlines)
    cleaned_text = text_without_extra_spaces.strip()
    # Append the cleaned text to the file, starting with a new line
    with open(filename, 'a', encoding='utf-8') as file:  # 'a' mode for appending
        file.write(cleaned_text + "\n")  # Add a newline at the end for each entry
    # print(f"Cleaned text appended to {filename}")
    return cleaned_text
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('static/images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0
def get_text(image_path):
  img = cv2.imread(image_path)
#   cv2_imshow(cv2.resize(img,(512,1024)))
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  text = pytesseract.image_to_string(gray_img, config='psm=12 oem=2')
  cleaned_text = remove_newlines_and_extra_spaces_and_save(text)
  return cleaned_text

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
#                                         input_shape=[], # shape of inputs coming to our model
#                                         dtype=tf.string, # data type of inputs coming to the USE layer
#                                         trainable=False, # keep the pretrained weights (we'll create a feature extractor)
#                                         name="USE")
# model= tf.keras.Sequential([
#   sentence_encoder_layer, # take in sentences and then encode them into an embedding
#   layers.Dense(128, activation="relu"),
#   layers.Dropout(0.1),
#   layers.Dense(64,activation="relu"),
#   layers.Dense(3, activation="softmax")
# ], name="model_6_USE")

# # Compile model
# # model.compile(loss="sparse_categorical_crossentropy",
#                 optimizer=tf.keras.optimizers.Adam(),
#                 metrics=["accuracy"])
# model.load_weights('use_model.h5')
print('loaded')
st.write(""" # My first app Hello *world!*""")
uploaded_file = st.file_uploader("Upload Your File Here!")

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image

        display_image = Image.open(uploaded_file)

        st.image(display_image)
        text = get_text(os.path.join('static/images',uploaded_file.name))
        lis = [text]
        y_pred = model.predict(np.array(lis))
        print(y_pred)
        class_names = ['Datsheet', 'PnID', 'EngDWG']
        prediction  = class_names[np.argmax(y_pred)]
        print(prediction)
        # prediction = model.predict(os.path.join('static/images',uploaded_file.name))
        os.remove('static/images/'+uploaded_file.name)
        # deleting uploaded saved picture after prediction

        # drawing graphs

        st.text('Predictions :-')

        fig, ax = plt.subplots()

        ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)

        ax.set(xlabel='Confidence %', ylabel='Breed')

        st.pyplot(fig)
# print(file)
