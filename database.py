import sqlite3
import os
import sys

# Define the database path
DB_PATH = 'face_auth.db'

# Function to initialize the database
def initialize_db():
    # Check if the database file exists
    if not os.path.exists(DB_PATH):
        # If not, create the database and set permissions
        try:
            # Check if the directory is writable
            if not os.access('.', os.W_OK):
                raise PermissionError("Current directory is not writable.")

            # Attempt to create the database file
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Create the Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    face_embedding BLOB NOT NULL
                )
            ''')

            # Create the Faces table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    face_embedding BLOB NOT NULL
                )
            ''')

            # Commit and close the connection
            conn.commit()
            conn.close()

            # Set the correct permissions (only creator can access)
            os.chmod(DB_PATH, 0o600)  # This gives read-write permissions only to the owner (creator)

            print("Database created successfully and permissions set.")
        
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            sys.exit(1)
        except PermissionError as e:
            print(f"Permission error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
    
    else:
        print(f"Database already exists at {DB_PATH}.")


def get_db_connection():
    """Secure connection to the database with restricted permissions."""
    return sqlite3.connect(DB_PATH)

initialize_db()
