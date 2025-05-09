# app.py 

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from database import connect_db, fetch_data
from utils import create_table_and_insert

# Streamlit UI
st.title("PyTorch MNIST Digit Recognizer")
st.write("Draw a digit (0-9) below and click Predict")

# Set up the drawing canvas
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

# Predict Button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        st.info("Prediction feature not implemented yet. Stay tuned!")
        img = canvas_result.image_data
        
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