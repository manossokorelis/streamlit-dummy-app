import streamlit as st
from database import connect_db, fetch_data
from utils import create_table_and_insert

# Streamlit UI
st.title("Digits Recognizer 3")
st.write("Draw a digit on the canvas:")

# Set up the drawing canvas
# canvas_result = st_canvas(
#     fill_color="white",  # background color
#     stroke_width=20,     # stroke width for drawing
#     stroke_color="black", # stroke color for drawing
#     background_color="white",  # canvas background color
#     width=280,
#     height=280,
#     drawing_mode="freedraw",  # allows freeform drawing
#     key="canvas",
# )

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