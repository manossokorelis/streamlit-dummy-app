# app.py 

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scripts.model import load_model
from scripts.utils import create_table, fetch_data, insert_prediction

# Page config
st.set_page_config(layout="centered")
st.title("PyTorch MNIST Digit Recognizer")
st.write("Draw a digit (0-9) below and click Predict.")
col1, col2 = st.columns(2)
if "prediction_clicked" not in st.session_state:
    st.session_state.prediction_clicked = False
if "last_canvas_data" not in st.session_state:
    st.session_state.last_canvas_data = None

# Create a table 
create_table()

# Load trained model
model = load_model("data/mnist_cnn.pth")
model.eval()

# ---- LEFT COLUMN ----
with col1:
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=20,
        stroke_color='#ffffff',
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode='freedraw',
        key="canvas",
    )
    if canvas_result.image_data is not None:
        current_img_data = canvas_result.image_data.copy()
        if st.session_state.last_canvas_data is not None and not np.array_equal(current_img_data, st.session_state.last_canvas_data):
            st.session_state.prediction_clicked = False  
        st.session_state.last_canvas_data = current_img_data
    if st.button("Predict"):
        st.session_state.prediction_clicked = True  

# ---- RIGHT COLUMN ----
with col2:
    if canvas_result.image_data is not None and st.session_state.get("prediction_clicked", False):
        img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            conf = F.softmax(output, dim=1)[0][pred].item() * 100
        st.metric(label="Prediction", value=str(pred), delta=f"{conf:.1f}%")
        with st.form("feedback_form"):
            true_label = st.number_input("Enter True Label:", min_value=0, max_value=9, step=1)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                insert_prediction(pred=pred, true_label=true_label, confidence=conf)

# ---- PREDICTION HISTORY ----
st.subheader("Prediction History")
data = fetch_data()
if data:
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    col1.markdown("**Timestamp**")
    col2.markdown("**Pred**")
    col3.markdown("**True**")
    col4.markdown("**Conf**")
    for row in data:
        timestamp = row[1]
        pred = row[2]
        true = row[3]
        conf = f"{float(row[4]):.0f}%"
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
        col1.markdown(f"{timestamp}")
        col2.markdown(f"{pred}")
        col3.markdown(f"{true}")
        col4.markdown(f"{conf}")
else:
    st.info("No prediction history to display.")

# ---- GITHUB ----
st.write("---")
st.write("Made by [manossokorelis](https://github.com/manossokorelis)")
st.write("Source code available at: [streamlit-dummy-app](https://github.com/manossokorelis/streamlit-dummy-app)")
