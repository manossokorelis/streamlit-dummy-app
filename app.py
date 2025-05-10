# app.py 

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import load_model
from database import create_table_and_insert, fetch_data 

# Page config
st.set_page_config(layout="centered")
st.title("PyTorch MNIST Digit Recognizer")
st.write("Draw a digit (0–9) below and click Predict")

# Load trained model
model = load_model("mnist_cnn.pth")
model.eval()

# Create 2 columns: Left for canvas, right for prediction
col1, col2 = st.columns(2)

# ---- LEFT COLUMN ----
with col1:
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=10,
        stroke_color='#ffffff',
        background_color="#000000",
        height=150,width=150,
        drawing_mode='freedraw',
        key="canvas",
    )
    if st.button("Predict"):
        st.session_state.prediction_clicked = True
    else:
        st.session_state.prediction_clicked = st.session_state.get("prediction_clicked", False)

# ---- RIGHT COLUMN ----
with col2:
    if canvas_result.image_data is not None and st.session_state.get("prediction_clicked", False):
        img = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize and reshape image
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            conf = F.softmax(output, dim=1)[0][pred].item() * 100

        st.markdown(f"### Prediction: **{pred}**")
        st.markdown(f"**Confidence:** {conf:.1f}%")

        # Save prediction to DB (optional)
        create_table_and_insert(pred=pred, confidence=conf)


# Create table and insert sample data (optional)
create_table_and_insert()

# ---- PREDICTION HISTORY ----
st.subheader("Prediction History")
data = fetch_data()
if data:
    history_data = [
        {
            "Timestamp": row[1],
            "Pred": row[2],
            "True": row[3],
            "Conf": f"{float(row[4]):.1f}%"
        }
        for row in reversed(data) 
    ]
    df = pd.DataFrame(history_data)
    st.table(df)
else:
    st.info("No prediction history to display.")
