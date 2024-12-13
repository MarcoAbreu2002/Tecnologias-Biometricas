# Standard Library Imports
import sqlite3
import time
import os
import smtplib
import string
import ssl
import threading
import logging
from math import sqrt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
import io

# GUI (Tkinter) Related Imports
from tkinter import Tk, Label, Entry, Button, StringVar, messagebox, Frame, simpledialog, Toplevel
from PIL import Image, ImageTk

# Data Handling and ML Libraries
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from io import BytesIO

# Security Imports
from security import generate_key, decrypt_data, verify_password, hash_password, encrypt_data

# API and Networking
import requests

# Image Processing
import cv2
import face_recognition


# Global idle timer
last_activity_time = time.time()
is_idle = False
is_logged = False
idle_timeout = 10  

current_user = None

# Database setup
db_path = "users.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
security_flag = True

# Create tables for users and keystroke data
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password BLOB NOT NULL,
    email TEXT NOT NULL,
    face_embedding BLOB
)
""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS keystrokes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    ht_mean REAL, ht_std REAL,
    ppt_mean REAL, ppt_std REAL,
    rrt_mean REAL, rrt_std REAL,
    rpt_mean REAL, rpt_std REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS models (
    user_id INTEGER PRIMARY KEY,
    model_blob BLOB
);

""")
conn.commit()

def reset_idle_timer(event=None):
    """Reset the idle timer on any user interaction."""
    global last_activity_time
    last_activity_time = time.time()

def check_idle_time():
    """Check if the user has been idle for too long and prompt for re-authentication."""
    global last_activity_time
    global is_idle
    while True:
        current_time = time.time()
        if current_time - last_activity_time > idle_timeout and is_idle == False and is_logged == True:
            is_idle = True
            print("User has been idle for 2 minutes. Prompting for authentication...")
            app.after(0, idle_prompt)  # Run on main thread to update GUI
            reset_idle_timer()
        time.sleep(1)

def idle_prompt():
    """Prompt the user for password and facial recognition after idle timeout."""
    messagebox.showinfo("Session Locked", "You've been idle for too long. Please re-authenticate.")
    # Call your existing facial recognition and password check function
    authenticated = loginAfterIDLE(current_user)
    if authenticated:
        messagebox.showinfo("Authencication completed.")
        show_home()
    else:
        messagebox.showerror("Error", "Face authentication failed. Session locked.")
        logout()  # Or redirect to login


def send_security_alert_in_background(user_email):
    thread = threading.Thread(target=send_security_alert, args=(user_email,))
    thread.daemon = True   
    thread.start()


EMAIL_SENDER = "bancosegurotb@gmail.com"
PASS_APP = "sssi dexj elyg guqu"

def send_security_alert(user_email):
    subject = "Suspicious Login Attempt Detected"
    body = """
        Dear User,

        We noticed a suspicious login attempt to your account. The keystroke data from the recent login does not match the pattern we have on file.

        If you did not initiate this login, please take immediate action to secure your account.

        Thank you,
        Your Security Team
        """
    em = EmailMessage()
    em['From'] = EMAIL_SENDER
    em['To'] = user_email
    em['Subject'] = subject
    em.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, PASS_APP)
        smtp.sendmail(EMAIL_SENDER, user_email, em.as_string())


def register_face(username, conn=None):
    """Register the user's face securely using their encrypted email."""
    # Ask for permission to access the camera
    response = messagebox.askquestion(
        "Face Authentication",
        "Do you allow the app to access your camera for facial recognition?"
    )
    if response != 'yes':
        messagebox.showerror("Error", "Camera access denied.")
        return False

    print("Please face the camera to register your face.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera. Check permissions.")
        return False

    registered = False
    retries = 0
    max_retries = 10  # Limit retries to avoid resource drain

    try:
        while not registered and retries < max_retries:
            ret, frame = cap.read()
            if not ret:
                print("Error accessing the webcam. Please try again.")
                retries += 1
                continue

            # Process the captured frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) == 1:
                # Encrypt and save the face embedding
                face_embedding = np.array(face_encodings[0], dtype=np.float64).tobytes()

                # Use the provided connection to avoid locking
                if conn is None:
                    conn = sqlite3.connect(db_path, timeout=10)

                cursor = conn.cursor()
                encryption_key = load_aes_key()
                cursor.execute(
                    'UPDATE users SET face_embedding=? WHERE username=?',
                    (sqlite3.Binary(face_embedding), username)
                )
                conn.commit()

                print("Face registered successfully!")
                messagebox.showinfo("Success", "Face registered successfully!")
                registered = True
            else:
                print("Ensure only your face is visible and retry.")
                retries += 1

            cv2.imshow("Register Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not registered:
            print("Face registration failed. Please try again.")
            messagebox.showerror("Error", "Face registration failed.")
            return False
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


def authenticate_face(username):
    """Authenticate the user's face during login with secure handling."""
    # Ask for permission to access the camera
    response = messagebox.askquestion("Face Authentication", "Do you allow the app to access your camera for facial recognition?")
    if response != 'yes':
        messagebox.showerror("Error", "Camera access denied.")
        return None

    print("Authenticating face... Please ensure your face is clearly visible.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera. Check permissions.")
        return None

    start_time = time.time()
    authenticated = False
    retries = 0
    max_retries = 10  # Limit retries for better resource management

    try:
        while not authenticated and retries < max_retries:
            elapsed_time = time.time() - start_time
            if elapsed_time > 8:
                print("Authentication timed out. No face recognized within 8 seconds.")
                cap.release()
                cv2.destroyAllWindows()
                return None

            ret, frame = cap.read()
            if not ret:
                print("Error accessing the webcam.")
                retries += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) == 1:
                # Retrieve and decrypt the stored face embedding
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute('SELECT face_embedding, password FROM users WHERE username = ?', (username,))
                result = cursor.fetchone()
                conn.close()

                if result and result[0]:
                    stored_embedding_blob = result[0]
                    stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float64)

                    # Compare the face embeddings using Euclidean distance
                    distance = np.linalg.norm(stored_embedding - face_encodings[0])
                    if distance < 0.6:  # Threshold for face matching
                        authenticated = True
                        print(f"Authentication successful for user: {username}")
                        cap.release()
                        cv2.destroyAllWindows()
                        return username  # Successfully authenticated, return the user's email
                else:
                    print("No face data found for this user.")
                    messagebox.showerror("Error", "No face data registered for this user.")

            else:
                print("Face not recognized. Please adjust your position.")

            cv2.imshow("Authenticate Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not authenticated:
            print("Authentication failed. Please try again.")
            messagebox.showerror("Error", "Authentication failed. Please try again.")
            return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()



# Separate keystroke data for password and confirm password
password_keystrokes = {"press_times": [], "release_times": []}
confirm_password_keystrokes = {"press_times": [], "release_times": []}

# Utility Functions
def calculate_mean_and_std(feature_list):
    if not feature_list:
        return 0, 0
    mean = sum(feature_list) / len(feature_list)
    squared_diffs = [(x - mean) ** 2 for x in feature_list]
    variance = sum(squared_diffs) / (len(feature_list) - 1 if len(feature_list) > 1 else 1)
    std_dev = sqrt(variance)
    return mean, std_dev

def compute_keystroke_features(keystroke_data):
    press_times = keystroke_data["press_times"]
    release_times = keystroke_data["release_times"]

    ht = [release_times[i] - press_times[i] for i in range(len(press_times))]
    ppt = [press_times[i + 1] - press_times[i] for i in range(len(press_times) - 1)]
    rrt = [release_times[i + 1] - release_times[i] for i in range(len(release_times) - 1)]
    rpt = [press_times[i + 1] - release_times[i] for i in range(len(release_times) - 1)]

    return {
        "ht_mean": calculate_mean_and_std(ht)[0],
        "ht_std": calculate_mean_and_std(ht)[1],
        "ppt_mean": calculate_mean_and_std(ppt)[0],
        "ppt_std": calculate_mean_and_std(ppt)[1],
        "rrt_mean": calculate_mean_and_std(rrt)[0],
        "rrt_std": calculate_mean_and_std(rrt)[1],
        "rpt_mean": calculate_mean_and_std(rpt)[0],
        "rpt_std": calculate_mean_and_std(rpt)[1],
    }

def verify_user(keystroke_data, user_id, threshold=0.75):
    print(f"Starting verification for user_id: {user_id}")
    print(f"Received keystroke data: {keystroke_data}")

    # Retrieve the model
    model = retrieve_user_model(user_id)
    if model is None:
        print("Model not found for the given user ID.")
        return False

    # Debugging: Ensure keystroke_data contains all required keys
    required_keys = [
        'ht_mean', 'ht_std', 'ppt_mean', 'ppt_std',
        'rrt_mean', 'rrt_std', 'rpt_mean', 'rpt_std'
    ]
    missing_keys = [key for key in required_keys if key not in keystroke_data]
    if missing_keys:
        print(f"Error: Missing required keys in keystroke data: {missing_keys}")
        raise ValueError(f"Missing keys: {missing_keys}")

    # Extract features
    features = [keystroke_data[key] for key in required_keys]
    print(f"Extracted features: {features}")

    # Predict probabilities and class
    try:
        probabilities = model.predict_proba([features])[0]   
        print(f"Prediction probabilities: {probabilities}")
        predicted_class = model.classes_[np.argmax(probabilities)]
        print(f"Predicted class: {predicted_class}")
        print(f"Probability of predicted class: {probabilities[np.argmax(probabilities)]}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

    # Verify against user ID and threshold
    if predicted_class == user_id and probabilities[np.argmax(probabilities)] >= threshold:
        print("Verification result: Match")
        return True
    else:
        print("Verification result: No match")
        return False

def predict_user_model(user_id, new_data, conn):
    """
    Predicts the user input using the stored model in the SQLite database.

    :param user_id: The user ID for which the model is stored in the database.
    :param new_data: A DataFrame containing the new data for prediction.
    :param conn: An existing SQLite connection object.
    :return: Prediction result for the given user_id and new_data.
    """
    try:
        # Retrieve the serialized model blob for the given user_id
        cursor = conn.cursor()
        cursor.execute("SELECT model_blob FROM models WHERE user_id=?", (user_id,))
        result = cursor.fetchone()

        if result is None:
            raise ValueError(f"No model found for user_id: {user_id}")

        # Deserialize the model from the BLOB data
        model_blob = result[0]
        model_stream = BytesIO(model_blob)
        rf_model = joblib.load(model_stream)

        # Ensure the new_data has the same feature columns as the training data
        features = ['ht_mean', 'ht_std_dev', 'ppt_mean', 'ppt_std_dev', 'rrt_mean', 'rrt_std_dev', 'rpt_mean', 'rpt_std_dev']
        if not all(feature in new_data.columns for feature in features):
            raise ValueError("Input data does not have the required features")

        # Prepare the features for prediction (assumes new_data is a DataFrame with correct columns)
        X_new = new_data[features]
        
        # Make predictions using the deserialized model
        predictions = rf_model.predict(X_new)
        
        return predictions

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e

def predict_user(keystroke_data, user_id, conn=None):
    """
    Predicts the user based on keystroke data using a model retrieved from the database.

    :param user_id: ID of the user for whom the prediction is made.
    :param keystroke_data: Raw keystroke data (press and release times).
    :param db_path: Path to the SQLite database file.
    :return: Predicted user ID.
    """
    # Compute keystroke features
    feature_vector = [
        keystroke_data["ht_mean"],
        keystroke_data["ht_std"],
        keystroke_data["ppt_mean"],
        keystroke_data["ppt_std"],
        keystroke_data["rrt_mean"],
        keystroke_data["rrt_std"],
        keystroke_data["rpt_mean"],
        keystroke_data["rpt_std"]
    ]
        # Use the provided connection to avoid locking
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=10)
    # Retrieve the model from the database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT model_blob FROM models WHERE user_id = ?""", (user_id,))
        result = cursor.fetchone()

    if not result:
        raise ValueError(f"No model found for user ID {user_id}.")

    # Deserialize the model using joblib
    try:
        model_stream = BytesIO(result[0])
        model = joblib.load(model_stream)
    except Exception as e:
        raise ValueError(f"Error deserializing the model for user ID {user_id}: {str(e)}")

    # Make a prediction using the retrieved model
    prediction = model.predict([feature_vector])

    # Return the predicted user ID
    return prediction[0]



def retrieve_user_model(user_id):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT model_blob FROM models WHERE user_id=?", (user_id,))
            row = cursor.fetchone()
            if row:
                model_blob = io.BytesIO(row[0])
                model = joblib.load(model_blob)
                return model
            else:
                print(f"No model found for user {user_id}.")
                return None
    except sqlite3.Error as e:
        print(f"Error loading model for user {user_id}: {e}")
        return None



def reset_keystroke_data():
    """Reset keystroke data after it's processed."""
    global password_keystrokes, confirm_password_keystrokes
    password_keystrokes["press_times"] = []
    password_keystrokes["release_times"] = []
    confirm_password_keystrokes["press_times"] = []
    confirm_password_keystrokes["release_times"] = []


def process_sqlite_data(user_id):
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch data from the keystrokes table
    cursor.execute("SELECT * FROM keystrokes WHERE user_id=?", (user_id,))
    rows = cursor.fetchall()

    # Prepare dictionary to store the data
    data = {
        'user_id': [],
        'ht_mean': [],
        'ht_std_dev': [],
        'ppt_mean': [],
        'ppt_std_dev': [],
        'rrt_mean': [],
        'rrt_std_dev': [],
        'rpt_mean': [],
        'rpt_std_dev': [],
    }

    # Process each row from the query result
    for row in rows:
        user_id = row[1]  # user_id is in the second column (index 1)
        ht_mean = row[2]
        ht_std_dev = row[3]
        ppt_mean = row[4]
        ppt_std_dev = row[5]
        rrt_mean = row[6]
        rrt_std_dev = row[7]
        rpt_mean = row[8]
        rpt_std_dev = row[9]

        # Append the calculated values to the data dictionary
        data['user_id'].append(user_id)
        data['ht_mean'].append(ht_mean)
        data['ht_std_dev'].append(ht_std_dev)
        data['ppt_mean'].append(ppt_mean)
        data['ppt_std_dev'].append(ppt_std_dev)
        data['rrt_mean'].append(rrt_mean)
        data['rrt_std_dev'].append(rrt_std_dev)
        data['rpt_mean'].append(rpt_mean)
        data['rpt_std_dev'].append(rpt_std_dev)

    # Convert the dictionary to a DataFrame
    data_df = pd.DataFrame(data)

    # Close the database connection
    conn.close()

    return data_df

logging.basicConfig(level=logging.INFO)

def train_model(user_id):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM keystrokes WHERE user_id=?", (user_id,))
            rows = cursor.fetchall()

            if not rows or len(rows) < 2:
                logging.warning(f"Not enough data to train the model for user_id {user_id}.")
                return None

            X = [row[2:10] for row in rows]  # Features
            y = [row[1] for row in rows]     # Labels (user_id)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")

            model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            # Log model performance
            if X_test:
                y_pred = model.predict(X_test)
                logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            else:
                logging.warning("No test set available for evaluation.")

            # Serialize and store the model
            model_blob = io.BytesIO()
            joblib.dump(model, model_blob)
            model_blob.seek(0)

            cursor.execute(""" 
                INSERT INTO models (user_id, model_blob)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET model_blob=excluded.model_blob
            """, (user_id, model_blob.read()))
            conn.commit()

            logging.info(f"Model for user_id {user_id} saved successfully.")
            return model

    except Exception as e:
        logging.error(f"Error during model training for user_id {user_id}: {e}")
        return None


def train_model2(training_data, user_id, db_path, conn=None):
    """
    Trains a Random Forest model on the given data and stores it in a SQLite database.

    :param training_data: DataFrame containing training features and user_id
    :param user_id: The user ID for which the model is being trained
    :param db_path: Path to the SQLite database file
    :param conn: SQLite connection object (optional)
    """
    features = ['ht_mean', 'ht_std_dev', 'ppt_mean', 'ppt_std_dev', 'rrt_mean', 'rrt_std_dev', 'rpt_mean', 'rpt_std_dev']
    X = training_data[features]
    y = training_data['user_id']  # Ensure this is the intended target variable

    # Train a Random Forest model on the data
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)

    # Serialize the model using joblib and store it in memory
    model_stream = BytesIO()
    joblib.dump(rf_model, model_stream)
    model_stream.seek(0)  # Reset stream position to the beginning
    serialized_model = model_stream.read()

    # Use the provided connection to avoid locking, if conn is not provided, create a new one
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=10)

    # Use a context manager to manage the SQLite connection
    try:
        cursor = conn.cursor()
        # Insert or update the model in the database
        cursor.execute("""
            INSERT INTO models (user_id, model_blob)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET model_blob=excluded.model_blob
        """, (user_id, serialized_model))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error storing the model: {e}")
        raise e

def on_key_press_password(event):
    """Record the timestamp when a key is pressed for the password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        password_keystrokes["press_times"].append(time.time())
        print(password_keystrokes)

def on_key_release_password(event):
    """Record the timestamp when a key is released for the password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        password_keystrokes["release_times"].append(time.time())
        print(password_keystrokes)

def on_key_press_confirm_password(event):
    """Record the timestamp when a key is pressed for the confirm password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        confirm_password_keystrokes["press_times"].append(time.time())
        print(confirm_password_keystrokes)

def on_key_release_confirm_password(event):
    """Record the timestamp when a key is released for the confirm password field."""
    # Skip modifiers (Shift, Control, Alt) and Backspace/Delete
    if event.keysym != "Tab" and event.keysym not in ["Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock", "BackSpace", "Delete"]:
        confirm_password_keystrokes["release_times"].append(time.time())
        print(confirm_password_keystrokes)

def load_aes_key(filename='aes_key.key'):
    with open(filename, 'rb') as key_file:
        return key_file.read()

def register_user():
    """Register a new user with improved security."""
    username = reg_username.get().strip()
    password = reg_password.get().strip()
    confirm_password = reg_password_confirm.get().strip()
    email = reg_email.get().strip()

    # Input validation
    if not username or not password or not confirm_password or not email:
        messagebox.showerror("Error", "All fields are required!")
        return

    if password != confirm_password:
        messagebox.showerror("Error", "Passwords do not match!")
        reset_keystroke_data()
        return

    if len(password) < 8:  # Password strength check
        messagebox.showerror("Error", "Password must be at least 8 characters long!")
        return

    # First password entry keystroke dynamics
    reg_password_entry.bind("<KeyPress>", on_key_press_password)
    reg_password_entry.bind("<KeyRelease>", on_key_release_password)
    features_1 = compute_keystroke_features(password_keystrokes)
    if any(v == 0 for v in features_1.values()):
        messagebox.showerror("Error", "Invalid keystroke data for the first input!")
        reset_keystroke_data()
        return

    # Confirm password keystroke dynamics
    reg_password_confirm_entry.bind("<KeyPress>", on_key_press_confirm_password)
    reg_password_confirm_entry.bind("<KeyRelease>", on_key_release_confirm_password)
    features_2 = compute_keystroke_features(confirm_password_keystrokes)
    if any(v == 0 for v in features_2.values()):
        messagebox.showerror("Error", "Invalid keystroke data for the second input!")
        reset_keystroke_data()
        return

    try:
        hashed_password = hash_password(password)

        # Encrypt email and username using the same salt
        encryption_key = load_aes_key()
        encrypted_email = encrypt_data(email, encryption_key)

        # Insert user details into the database (transactional approach)
        with sqlite3.connect(db_path, timeout=10) as conn:
            cursor = conn.cursor()

            # Start a transaction
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                (username, hashed_password, encrypted_email)
            )
            user_id = cursor.lastrowid

            # Insert keystroke data for both entries
            cursor.execute("""
                INSERT INTO keystrokes (user_id, ht_mean, ht_std, ppt_mean, ppt_std, rrt_mean, rrt_std, rpt_mean, rpt_std)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, *features_1.values()))
            cursor.execute("""
                INSERT INTO keystrokes (user_id, ht_mean, ht_std, ppt_mean, ppt_std, rrt_mean, rrt_std, rpt_mean, rpt_std)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, *features_2.values()))

            # Attempt to register face
            print(f"Registering face for user {username}.")
            face_registration_success = register_face(username, conn)

            if not face_registration_success:
                # Rollback if face registration fails
                print("Face registration failed, rolling back user registration.")
                conn.rollback()
                messagebox.showerror("Error", "Face registration failed. User registration aborted.")
                return

            # Commit the transaction if everything succeeded
            conn.commit()

        # Process and retrain the model after a successful registration
        training_data = process_sqlite_data(user_id)
        train_model2(training_data, user_id, conn)

        messagebox.showinfo("Success", "Registration successful!")
        reset_keystroke_data()
        show_login()

    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Username or email already exists!")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        reset_keystroke_data()


def login_user():
    """Handle user login with password, facial recognition, and keystroke verification."""
    global current_user, is_logged
    username = login_username.get().strip()
    password = login_password.get().strip()

    if not username or not password:
        messagebox.showerror("Error", "Both fields are required!")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use encrypted username for the query
            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cursor.fetchone()
    except sqlite3.IntegrityError:
        print("Error accessing the database")

    if user:
        encryption_key = load_aes_key()
        # Step 1: Authenticate face before checking password
        print("Face authentication started...")
        decrypted_email = decrypt_data(user[3], encryption_key)  # Decrypt email
        if not authenticate_face(username):  # Assuming face is registered with decrypted email
            messagebox.showerror("Error", "Face authentication failed.")
            return

        # Step 2: Check password with bcrypt hash
        hashed_password = user[2]   
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode()  # Convert to bytes
        if verify_password(password, hashed_password):  # Use bcrypt to verify password
            print("Password matched. Now checking keystroke data...")

            # Get the user email and user_id
            user_email = decrypted_email
            user_id = user[0]
            current_user = username

            #send_security_alert_in_background(user_email)
            # First password entry keystroke dynamics
            login_password_entry.bind("<KeyPress>", on_key_press_password)
            login_password_entry.bind("<KeyRelease>", on_key_release_password)
            features_1 = compute_keystroke_features(password_keystrokes)
            if any(v == 0 for v in features_1.values()):
                messagebox.showerror("Error", "Invalid keystroke data for the first input!")
                reset_keystroke_data()
                return
            # Assuming the first record contains the features
            #matched = verify_user(features_1, user_id)
            matched = predict_user_model(features_1, user_id, conn)
            if matched:
                print("There was a match")
            # Optionally, compare features for consistency here (e.g., keystroke analysis)

                try:
                    # Insert keystroke data
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(""" 
                            INSERT INTO keystrokes (user_id, ht_mean, ht_std, ppt_mean, ppt_std, rrt_mean, rrt_std, rpt_mean, rpt_std)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (user_id, *features_1.values()))
                        conn.commit()
                    # Retrain the model
                    training_data = process_sqlite_data(user_id)
                    train_model2(training_data, user_id, conn)
                    #train_model(user_id)
                    #reset_keystroke_data()
                except sqlite3.IntegrityError:
                    messagebox.showerror("Error", "Failed to insert keystroke data!")

                reset_keystroke_data()
                is_logged = True
                show_home(username)  # Proceed to the home screen if login is successful
            else:
                security_flag = True
                print("keystroke did not match, not saving and turning on security.")
        else:
            messagebox.showerror("Error", "Incorrect password.")
    else:
        messagebox.showerror("Error", "Username not found.")


def loginAfterIDLE(username):
    """
    Handle user login after idle with facial recognition, password, and keystroke verification.
    
    Args:
        email (str): The email of the user attempting to log in.
    """
    global is_idle, is_logged
    # Fetch the user details from the database using the provided email
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cursor.fetchone()
    except sqlite3.IntegrityError:
        print("Error accessing the database")
        return

    if not user:
        messagebox.showerror("Error", "User not found.")
        return

    # Create a new dialog for re-entering the password
    idle_login_window = Tk()
    idle_login_window.title("Re-Authenticate")
    idle_login_window.geometry("400x200")

    Label(idle_login_window, text="Please re-enter your password:").pack(pady=10)
    
    idle_password = StringVar()
    idle_password_entry = Entry(idle_login_window, textvariable=idle_password, show="*")
    idle_password_entry.pack(pady=5)
    idle_password_entry.focus()

    # Initialize keystroke data for the idle login
    idle_keystrokes = {"press_times": [], "release_times": []}

    def on_key_press_idle(event):
        if event.keysym != "Tab":  # Skip recording for Tab key
            idle_keystrokes["press_times"].append(time.time())
            print(idle_keystrokes)

    def on_key_release_idle(event):
        if event.keysym != "Tab":  # Skip recording for Tab key
            idle_keystrokes["release_times"].append(time.time())
            print(idle_keystrokes)

    # Attach keystroke listeners
    idle_password_entry.bind("<KeyPress>", on_key_press_idle)
    idle_password_entry.bind("<KeyRelease>", on_key_release_idle)

    def handle_idle_login():
        entered_password = idle_password_entry.get()
        hashed_password = user[2]   
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode()  # Convert to bytes

        if not entered_password:
            messagebox.showerror("Error", "Password input is required.")
            return

        # Authenticate face
        print("Face authentication started...")
        if not authenticate_face(username):
            messagebox.showerror("Error", "Face authentication failed.")
            return

        # Verify password
        if not verify_password(entered_password, hashed_password):  # Use bcrypt to verify password
            messagebox.showerror("Error", "Incorrect password.")
            show_login()

        is_idle = False

        # Compute keystroke features after user completes typing
        features = compute_keystroke_features(idle_keystrokes)
        if any(v == 0 for v in features.values()):
            messagebox.showerror("Error", "Invalid keystroke data for the input!")
            return

        # Fetch stored keystroke data
        cursor.execute("SELECT * FROM keystrokes WHERE user_id=?", (user[0],))
        keystroke_data = cursor.fetchone()

        if not keystroke_data:
            messagebox.showerror("Error", "No keystroke data found for this user.")
            return

        # Compare recorded keystroke data with stored data
        db_features = keystroke_data[2:]  # Assuming keystroke features start at column 2
        security_flag = False
        for i, key in enumerate(features):
            if abs(features[key] - db_features[i]) > 0.1:  # Threshold for keystroke similarity
                security_flag = True
                break

        if security_flag:
            encryption_key = load_aes_key()
            decrypted_email = decrypt_data(user[3], encryption_key)  # Decrypt email
            send_security_alert_in_background(decrypted_email)
            messagebox.showwarning("Warning", "Keystroke data does not match. An email has been sent to your address.")
            idle_login_window.destroy()
            return

        # Successful authentication
        print(f"Login successful for user: {username}")
        idle_login_window.destroy()
        show_home(username)  

    # Login button waits for user's action
    Button(idle_login_window, text="Login", command=handle_idle_login).pack(pady=20)
    idle_login_window.mainloop()

# Security validation function
def validate_physical_matrix(callback):
    security_prompt = Toplevel(app)
    security_prompt.title("Security Check")
    security_prompt.geometry("300x200")
    Label(security_prompt, text="Enter Physical Matrix", font=font_medium).pack(pady=10)
    matrix_input = Entry(security_prompt, font=font_medium)
    matrix_input.pack(pady=10)
    
    def submit_matrix():
        # Placeholder for validation logic
        if matrix_input.get() == "1234": 
            security_prompt.destroy()
            callback()
        else:
            Label(security_prompt, text="Invalid Matrix!", font=font_small, fg="red").pack()
    
    Button(security_prompt, text="Submit", command=submit_matrix, bg=button_color, fg="white").pack(pady=10)

# Function to create new pages dynamically
def show_page(page_title, input_fields):
    def render_page():
        if security_flag:
            validate_physical_matrix(render_inputs)
        else:
            render_inputs()
    
    def render_inputs():
        page = Toplevel(app)
        page.title(page_title)
        page.geometry("400x300")
        Label(page, text=page_title, font=font_large).pack(pady=10)
        for field in input_fields:
            Label(page, text=field, font=font_medium).pack(pady=5)
            Entry(page, font=font_small).pack(pady=5)
        Button(page, text="Submit", bg=button_color, fg="white", font=font_medium).pack(pady=20)
    
    render_page()

# Dynamic navigation to banking task pages
def open_view_balance():
    show_page("View Balance", ["Account Number"])

def open_transfer_funds():
    show_page("Transfer Funds", ["From Account", "To Account", "Amount"])

def open_transaction_history():
    show_page("Transaction History", ["Account Number", "Date Range"])


# Functionality for page switching
def show_register():
    # Clear registration fields
    reg_username.set("")
    reg_password.set("")
    reg_password_confirm.set("")
    reg_email.set("")
    reset_keystroke_data()
    
    login_frame.pack_forget()
    home_frame.pack_forget()
    reg_frame.pack()

def show_login():
    # Clear login fields
    login_username.set("")
    login_password.set("")
    reset_keystroke_data()
    reg_frame.pack_forget()
    home_frame.pack_forget()
    login_frame.pack()

def show_home(username):
    global is_idle
    is_idle = False

    def render_home():
        reset_keystroke_data()
        reg_frame.pack_forget()
        login_frame.pack_forget()
        home_label.config(text=f"Welcome {username}!")
        home_frame.pack()

    render_home()

def logout():
    global current_user, is_logged
    reg_username.set("")
    reg_password.set("")
    reg_password_confirm.set("")
    reg_email.set("")
    login_username.set("")
    login_password.set("")
    current_user = None
    is_logged = False
    
    # Destroy any dynamic pages if they are open
    for widget in app.winfo_children():
        if isinstance(widget, Toplevel):  # Close all Toplevel windows
            widget.destroy()

    # Return to the login screen
    show_login()

# GUI setup
app = Tk()
app.title("Bank App")
app.geometry("800x600")  # Enlarged window size
app.configure(bg="#FFFFFF")  # Bank app-themed background color

# Resize the logo using Pillow
original_logo = Image.open("logo.png")  
resized_logo = original_logo.resize((150, 100))  # (width, height)
logo_img = ImageTk.PhotoImage(resized_logo)

# Add the resized logo
logo_label = Label(app, image=logo_img, bg="#FFFFFF")
logo_label.place(x=10, y=10)  # Positioning the logo at the top left corner

# Updated fonts for larger text
font_large = ("Helvetica", 20, "bold")  # Larger font
font_medium = ("Helvetica", 16)
font_small = ("Helvetica", 14)

# Updated button styles for a polished look
primary_color = "#0d47a1"
button_color = "#1565c0"
button_style = {
    "bg": button_color,
    "fg": "white",
    "font": font_medium,
    "relief": "solid",
    "bd": 2
}


# Variables
reg_username = StringVar()
reg_password = StringVar()
reg_password_confirm = StringVar()
reg_email = StringVar()

login_username = StringVar()
login_password = StringVar()

def disable_copy_paste(event):
    return "break"  # Prevent the event from propagating
# Updated Registration Frame
reg_frame = Frame(app, bg="#FFFFFF")
Label(reg_frame, text="Register", font=font_large, bg="#FFFFFF").pack(pady=20)
Label(reg_frame, text="Username:", font=font_medium, bg="#FFFFFF").pack()
Entry(reg_frame, textvariable=reg_username, font=font_medium).pack(pady=5)
Label(reg_frame, text="Password:", font=font_medium, bg="#FFFFFF").pack()
reg_password_entry = Entry(reg_frame, textvariable=reg_password, show="*", font=font_medium)
# Disable copy and paste in the registration password field
reg_password_entry.bind("<Control-c>", disable_copy_paste)
reg_password_entry.bind("<Control-v>", disable_copy_paste)
reg_password_entry.bind("<Button-3>", disable_copy_paste)  # Right-click disable
reg_password_entry.pack(pady=5)
reg_password_entry.bind("<KeyPress>", on_key_press_password)
reg_password_entry.bind("<KeyRelease>", on_key_release_password)
Label(reg_frame, text="Confirm Password:", font=font_medium, bg="#FFFFFF").pack()
reg_password_confirm_entry = Entry(reg_frame, textvariable=reg_password_confirm, show="*", font=font_medium)
# Disable copy and paste in the confirmation password field
reg_password_confirm_entry.bind("<Control-c>", disable_copy_paste)
reg_password_confirm_entry.bind("<Control-v>", disable_copy_paste)
reg_password_confirm_entry.bind("<Button-3>", disable_copy_paste)  # Right-click disable
reg_password_confirm_entry.pack(pady=5)
reg_password_confirm_entry.bind("<KeyPress>", on_key_press_confirm_password)
reg_password_confirm_entry.bind("<KeyRelease>", on_key_release_confirm_password)
Label(reg_frame, text="Email:", font=font_medium, bg="#FFFFFF").pack()
Entry(reg_frame, textvariable=reg_email, font=font_medium).pack(pady=5)
Button(reg_frame, text="Register", command=register_user, **button_style).pack(pady=15)
Button(reg_frame, text="Go to Login", command=show_login, **button_style).pack()

# Updated Login Frame
login_frame = Frame(app, bg="#FFFFFF")
Label(login_frame, text="Login", font=font_large, bg="#FFFFFF").pack(pady=20)
Label(login_frame, text="Username:", font=font_medium, bg="#FFFFFF").pack()
Entry(login_frame, textvariable=login_username, font=font_medium).pack(pady=5)
Label(login_frame, text="Password:", font=font_medium, bg="#FFFFFF").pack()
login_password_entry = Entry(login_frame, textvariable=login_password, show="*", font=font_medium)
# Disable copy and paste in the login password field
login_password_entry.bind("<Control-c>", disable_copy_paste)
login_password_entry.bind("<Control-v>", disable_copy_paste)
login_password_entry.bind("<Button-3>", disable_copy_paste)  # Right-click disable
login_password_entry.pack(pady=5)
login_password_entry.bind("<KeyPress>", on_key_press_password)
login_password_entry.bind("<KeyRelease>", on_key_release_password)
Button(login_frame, text="Login", command=login_user, **button_style).pack(pady=15)
Button(login_frame, text="Go to Register", command=show_register, **button_style).pack(pady=5)
Button(login_frame, text="Exit", command=app.quit, bg="#e53935", fg="white", font=font_medium, relief="solid", bd=2).pack(pady=10)

# Updated Home Frame
home_frame = Frame(app, bg="#FFFFFF")
home_label = Label(home_frame, text="", font=font_large, bg="#FFFFFF")
home_label.pack(pady=20)
Button(home_frame, text="View Balance", command=open_view_balance, **button_style).pack(pady=10)
Button(home_frame, text="Transfer Funds", command=open_transfer_funds, **button_style).pack(pady=10)
Button(home_frame, text="Transaction History", command=open_transaction_history, **button_style).pack(pady=10)
Button(home_frame, text="Logout", command=logout, bg="#e53935", fg="white", font=font_medium, relief="solid", bd=2).pack(pady=10)



# Bind events to reset the idle timer
app.bind_all("<Any-KeyPress>", reset_idle_timer)
app.bind_all("<Any-Button>", reset_idle_timer)
app.bind_all("<Motion>", reset_idle_timer)

idle_thread = threading.Thread(target=check_idle_time, daemon=True)
idle_thread.start()

# Start with Login Frame
show_login()

app.mainloop()