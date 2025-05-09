# app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from database import connect_db, fetch_data
from utils import create_table_and_insert

# Page config
st.set_page_config(layout="wide")
st.title("PyTorch MNIST Digit Recognizer")
st.write("Draw a digit (0–9) below and click Predict")

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
    if st.session_state.get("prediction_clicked", False):
        st.metric(label="Prediction", value="N/A", delta="..%")
        # ---- Show feedback form after Predict ----
        with st.form("feedback_form"):
            true_label = st.number_input("Enter True Label:", min_value=0, max_value=9, step=1)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                st.success(f"Feedback logged to database!")


# Create table and insert sample data (optional)
create_table_and_insert()

# ---- PREDICTION HISTORY ----
st.subheader("Prediction History")

# Fetch prediction history from the database
data = fetch_data()

if data:
    # Build HTML table manually
    table_html = """
    <table style='width:100%; border-collapse: collapse;'>
        <thead>
            <tr>
                <th style='text-align: left; padding: 6px;'>Timestamp</th>
                <th style='text-align: left; padding: 6px;'>Pred</th>
                <th style='text-align: left; padding: 6px;'>True</th>
                <th style='text-align: left; padding: 6px;'>Conf</th>
            </tr>
        </thead>
        <tbody>
    """

    for row in data:
        timestamp = row[1]
        pred = row[2]
        true = row[3]
        conf = f"{float(row[4]):.1f}%"
        table_html += f"""
            <tr>
                <td style='padding: 6px;'>{timestamp}</td>
                <td style='padding: 6px;'>{pred}</td>
                <td style='padding: 6px;'>{true}</td>
                <td style='padding: 6px;'>{conf}</td>
            </tr>
        """

    table_html += "</tbody></table>"

    # Render HTML table in Streamlit
    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.info("No prediction history to display.")