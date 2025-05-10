import streamlit as st
from streamlit_drawable_canvas import st_canvas
from database import connect_db, fetch_data
from utils import create_table_and_insert
import pandas as pd
import torch
import numpy as np
from PIL import Image
from model import load_model  # Assuming model.py exists

# Page config
st.set_page_config(layout="centered")
st.title("PyTorch MNIST Digit Recognizer")
st.write("Draw a digit (0–9) below and click Predict")

# Load trained model
model = load_model("mnist_cnn.pth")

# Create 2 columns: Left for canvas, right for prediction
col1, col2 = st.columns(2)

# ---- LEFT COLUMN ----
with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Predict"):
        st.session_state.prediction_clicked = True
    else:
        st.session_state.prediction_clicked = st.session_state.get("prediction_clicked", False)

# ---- RIGHT COLUMN ----
with col2:
    if st.session_state.get("prediction_clicked", False) and canvas_result.image_data is not None:
        img = canvas_result.image_data

        # Preprocess image
        img = Image.fromarray((img[:, :, 0] * 255).astype(np.uint8))
        img = img.resize((28, 28)).convert("L")
        img = np.array(img)
        # img = 255 - img  # Invert
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()
            conf = torch.softmax(output, dim=1)[0][pred].item() * 100

        st.metric(label="Prediction", value=str(pred), delta=f"{conf:.1f}%")

        # Optional: Feedback form
        with st.form("feedback_form"):
            true_label = st.number_input("Enter True Label:", min_value=0, max_value=9, step=1)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                st.success("Feedback logged to database!")
                # Optional: Save to DB here

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
        for row in reversed(data)  # Show latest first
    ]
    df = pd.DataFrame(history_data)
    st.table(df)
else:
    st.info("No prediction history to display.")
