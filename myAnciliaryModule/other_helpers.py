import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import logging
import os

import sqlite3

def print_hello_world():
    """Prints 'Hello, World!' to the console."""
    print("Hello, World!")



def fetch_data_as_dataframe(query, db_path):
    try:
        # Connect to the database
        with sqlite3.connect(db_path) as conn:
            # Use pandas to execute the query and return the result as a DataFrame
            df = pd.read_sql_query(query, conn)
            return df
    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")
        return None
    