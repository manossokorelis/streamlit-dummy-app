import streamlit as st
from database import connect_db, fetch_data
from utils import create_table_and_insert
from streamlit_drawable_canvas import st_canvas

# Streamlit UI
st.title("Digits Recognizer 2")
st.write("This is a dummy Streamlit app deployed using Docker and Render.")

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

# Canvas for drawing digits
st.write("Draw a digit on the canvas:")

# Set up the drawing canvas
canvas_result = st_canvas(
    fill_color="white",  # background color
    stroke_width=20,     # stroke width for drawing
    stroke_color="black", # stroke color for drawing
    background_color="white",  # canvas background color
    width=280,
    height=280,
    drawing_mode="freedraw",  # allows freeform drawing
    key="canvas",
)

# Button to trigger the recognition process
if st.button("Recognize Digit"):
    if canvas_result.image_data is not None:
        # You can process the canvas data here for digit recognition
        # For example, convert the canvas into an image and pass it to a model
        st.write("Processing the drawn digit...")
        # Insert digit recognition model prediction code here
        # For demonstration purposes, you can just show the raw image:
        st.image(canvas_result.image_data, caption="Digit Drawn")
    else:
        st.warning("Please draw a digit on the canvas first!")
