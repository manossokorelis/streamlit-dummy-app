import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import load_model, predict_digit
from database import connect_db, fetch_data
from utils import create_table_and_insert

# Load the pre-trained model
model = load_model()

# Streamlit UI
st.title("MNIST Digits Recognizer")
st.write("Draw a digit (0-9) and click Predict to recognize it.")

# Set up the canvas for drawing
canvas_result = st_canvas(
    fill_color="white",
    stroke_color="black",
    stroke_width=15,
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Button to predict the drawn digit
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert the drawn image to PIL image
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))

        # Process the image (transform to tensor)
        image = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
            transforms.Resize((28, 28)),  # Resize to 28x28 as expected by MNIST model
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])(image)

        # Add batch dimension (required for PyTorch models)
        image = image.unsqueeze(0)

        # Make prediction
        prediction = predict_digit(model, image)

        # Display prediction result
        st.write(f"Predicted Digit: {prediction}")

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