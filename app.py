# app.py 

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from database import connect_db, fetch_data
from utils import create_table_and_insert

# Page config
st.set_page_config(layout="wide")
st.title("PyTorch MNIST Digit Recognizer")

# Create 2 columns: Left for canvas, right for prediction
col1, col2 = st.columns(2)

# ---- LEFT COLUMN ----
with col1:
    st.subheader("Draw a digit (0-9)")

    # Drawing canvas
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

    # Predict button
    if st.button("Predict"):
        st.session_state.prediction_clicked = True
    else:
        st.session_state.prediction_clicked = st.session_state.get("prediction_clicked", False)

# ---- RIGHT COLUMN ----
with col2:
    st.subheader("Prediction")

    if st.session_state.get("prediction_clicked", False):
        st.metric(label="Predicted Digit", value="N/A", delta="Coming Soon")
        st.info("Prediction logic not implemented yet. Stay tuned!")
    else:
        st.write("Click 'Predict' to see the result.")

# Create table and insert rows if not already done
create_table_and_insert()

# Fetch and display data from the database
data = fetch_data()
if data:
    st.write("Data from PostgreSQL database:")
    for row in data:
        st.write(row)
else:
    st.write("No data to display.")