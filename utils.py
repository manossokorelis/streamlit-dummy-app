# utils.py

import streamlit as st
from database import connect_db
from datetime import datetime
from psycopg2 import sql

# Create a table if it doesn't exist
def create_table():
    connection = connect_db()
    if connection:
        cursor = connection.cursor()
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            predicted_digit INTEGER,
            true_label INTEGER,
            confidence FLOAT
        );
        '''
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        connection.close()

# Query the database
def fetch_data():
    connection = connect_db()
    if connection:
        cursor = connection.cursor()
        query = "SELECT * FROM predictions LIMIT 10;" 
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
        query = sql.SQL("INSERT INTO predictions (timestamp, predicted_digit, true_label, confidence) VALUES (%s, %s, %s, %s);")
        cursor.execute(query, (timestamp, pred, true_label, confidence))
        connection.commit()
        cursor.close()
        connection.close()
        st.success("Feedback logged to database!")
    else:
        st.error("Error saving feedback to the database.")