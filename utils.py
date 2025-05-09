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
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100)
        );
        '''
        cursor.execute(create_table_query)
        connection.commit()
        
        # Insert rows into the table
        insert_query = '''
        INSERT INTO users (name, email)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING;  -- Prevent inserting duplicate rows
        '''
        rows_to_insert = [
            ('John Doe', 'john.doe@example.com'),
            ('Jane Smith', 'jane.smith@example.com'),
            ('Alice Johnson', 'alice.johnson@example.com')
        ]
        cursor.executemany(insert_query, rows_to_insert)
        connection.commit()
        
        # Close cursor and connection
        cursor.close()
        connection.close()
