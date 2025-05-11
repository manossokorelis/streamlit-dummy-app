# database.py

import psycopg2
import streamlit as st
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