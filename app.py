import streamlit as st
from database import connect_db, fetch_data
from utils import create_table_and_insert

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