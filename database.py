# database.py

import psycopg2
import streamlit as st
from psycopg2 import sql
import os
from datetime import datetime

# Set up the connection to the database using environment variables or directly
DB_HOST = "dpg-d0f2ofs9c44c738ktsr0-a.oregon-postgres.render.com"
DB_NAME = "myapp_db_0wgj"
DB_USER = "myapp_user"
DB_PASSWORD = "WzkkxzpJaegWJ3UwHQI0lt3dxECiJTAR"

# Establish the connection to the PostgreSQL database
def connect_db():
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port="5432"
        )
        return connection
    except Exception as e:
        st.error(f"Error: Unable to connect to the database - {e}")
        return None

# Query the database
def fetch_data():
    connection = connect_db()
    if connection:
        cursor = connection.cursor()
        query = "SELECT * FROM predictions;"  # Fetch data from the users table
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return rows
    else:
        return []

# Insert new prediction data
def insert_prediction(pred, true_label, confidence):
    connection = connect_db()
    if connection:
        cursor = connection.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = sql.SQL("INSERT INTO predictions (timestamp, predicted_label, true_label, confidence) VALUES (%s, %s, %s, %s);")
        cursor.execute(query, (timestamp, pred, true_label, confidence))
        connection.commit()
        cursor.close()
        connection.close()
        st.success("Prediction and feedback saved to database!")
    else:
        st.error("Error saving feedback to the database.")