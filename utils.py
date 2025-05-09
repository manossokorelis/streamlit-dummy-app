# utils.py

import psycopg2
import streamlit as st
from database import connect_db

# Create a table and insert some rows if the table doesn't exist
def create_table_and_insert():
    connection = connect_db()
    if connection:
        cursor = connection.cursor()
        
        # Create table if it doesn't exist
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
        
        # Insert a new prediction record (example)
        insert_query = '''
        INSERT INTO predictions (predicted_digit, true_label, confidence)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING;  -- Prevent inserting duplicate rows
        '''
        
        # You can update this part to insert real values dynamically after a prediction
        rows_to_insert = [
            (3, 1, 99.6),
            (3, 1, 99.6),
            (3, 1, 99.6)
        ]
        cursor.executemany(insert_query, rows_to_insert)
        connection.commit()
        
        # Close cursor and connection
        cursor.close()
        connection.close()
