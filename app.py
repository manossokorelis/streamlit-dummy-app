import streamlit as st
import psycopg2
from psycopg2 import sql
import os

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

# Query the database
def fetch_data():
    connection = connect_db()
    if connection:
        cursor = connection.cursor()
        query = "SELECT * FROM users LIMIT 5;"  # Fetch data from the users table
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        return rows
    else:
        return []

# Streamlit UI
st.title("Digits Recognizer")
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