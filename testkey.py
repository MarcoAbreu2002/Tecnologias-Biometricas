import time
import sqlite3
import numpy as np
from pynput import keyboard
import cv2
import face_recognition
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import hashlib

# Database settings
DB_NAME = "multi_auth.db"
SENTENCE = "The quick brown fox jumps over the lazy dog."

# Initialize the database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            face_embedding BLOB,
            avg_dwell_time REAL,
            avg_flight_time REAL
        )
    ''')
    conn.commit()
    conn.close()

# Face registration
def register_face(username):
    print("Please face the camera to register your face.")
    cap = cv2.VideoCapture(0)
    registered = False

    while not registered:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the webcam. Please try again.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print(f"Detected {len(face_encodings)} face(s).")
        if len(face_encodings) == 1:
            face_embedding = face_encodings[0].tobytes()
            print(f"Face embedding length: {face_embedding}")

            # Save to database
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO users (username, face_embedding) VALUES (?, ?)', 
                    (username, sqlite3.Binary(face_embedding))
                )
                conn.commit()
                conn.close()
                registered = True
                print("Face registered successfully!")
            except Exception as e:
                print(f"Error inserting into database: {e}")
        else:
            print("Please ensure only your face is visible.")

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Keystroke dynamics registration
def register_keystroke(username):
    print("Please type the following sentence exactly as shown:\n")
    print(SENTENCE)
    dwell_times = []
    flight_times = []

    def on_press(key):
        try:
            key_times[key.char] = time.time()
        except AttributeError:
            key_times[str(key)] = time.time()

    def on_release(key):
        nonlocal dwell_times, flight_times
        try:
            press_time = key_times.get(key.char)
            if press_time:
                dwell_time = time.time() - press_time
                dwell_times.append(dwell_time)
        except AttributeError:
            press_time = key_times.get(str(key))
            if press_time:
                dwell_time = time.time() - press_time
                dwell_times.append(dwell_time)

        # Calculate flight time only if enough key events exist
        if len(key_times) > 1:
            last_two_keys = list(key_times.keys())[-2:]
            if all(k in key_times for k in last_two_keys):  # Ensure both keys exist
                flight_time = key_times[last_two_keys[1]] - key_times[last_two_keys[0]]
                flight_times.append(flight_time)

        # Stop the listener when Enter is pressed
        if key == keyboard.Key.enter:
            print("Enter key pressed. Stopping listener.")
            return False

    key_times = {}
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    if len(dwell_times) > 1:
        avg_dwell_time = np.mean(dwell_times)
        avg_flight_time = np.mean(flight_times)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO users (username, avg_dwell_time, avg_flight_time)
            VALUES (?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET avg_dwell_time = excluded.avg_dwell_time,
                                                 avg_flight_time = excluded.avg_flight_time
        ''', (username, avg_dwell_time, avg_flight_time))
        conn.commit()
        conn.close()

        print("Keystroke data registered successfully!")
    else:
        print("Not enough keystrokes to register. Please try again.")


# Real-time face authentication
def authenticate_face(username):
    print("Authenticating face...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT face_embedding FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()

    if not result:
        print("Face data not found. Authentication failed.")
        return False

    stored_embedding = np.frombuffer(result[0], dtype=np.float64)

    cap = cv2.VideoCapture(0)
    authenticated = False

    while not authenticated:
        ret, frame = cap.read()
        if not ret:
            print("Error accessing the webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 1:
            distance = np.linalg.norm(stored_embedding - face_encodings[0])
            if distance < 0.6:
                authenticated = True
                print("Face authentication successful!")
                break
        else:
            print("Face not recognized. Please try again.")

        cv2.imshow("Authenticate Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return authenticated

from pynput import keyboard

# Keystroke authentication
def authenticate_keystroke(username):
    print("Please type the following sentence exactly as shown:\n")
    print(SENTENCE)
    dwell_times = []
    flight_times = []

    def on_press(key):
        try:
            key_times[key.char] = time.time()
        except AttributeError:
            key_times[str(key)] = time.time()

    def on_release(key):
        nonlocal dwell_times, flight_times
        try:
            press_time = key_times.get(key.char)
            if press_time:
                dwell_time = time.time() - press_time
                dwell_times.append(dwell_time)
        except AttributeError:
            press_time = key_times.get(str(key))
            if press_time:
                dwell_time = time.time() - press_time
                dwell_times.append(dwell_time)

        # Calculate flight time only if enough key events exist
        if len(key_times) > 1:
            last_two_keys = list(key_times.keys())[-2:]
            if all(k in key_times for k in last_two_keys):  # Ensure both keys exist
                flight_time = key_times[last_two_keys[1]] - key_times[last_two_keys[0]]
                flight_times.append(flight_time)

        # Stop the listener when Enter is pressed
        if key == keyboard.Key.enter:
            print("Enter key pressed. Stopping listener.")
            return False

    key_times = {}
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    if len(dwell_times) > 1:
        avg_dwell_time = np.mean(dwell_times)
        avg_flight_time = np.mean(flight_times)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT avg_dwell_time, avg_flight_time FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()

        if result:
            stored_dwell_time, stored_flight_time = result
            if abs(avg_dwell_time - stored_dwell_time) < 0.05 and abs(avg_flight_time - stored_flight_time) < 0.05:
                print("Keystroke authentication successful!")
                return True
            else:
                print("Typing pattern does not match. Authentication failed.")
                return False
        else:
            print("No keystroke data found. Authentication failed.")
            return False
    else:
        print("Not enough keystrokes. Authentication failed.")
        return False


# Main system
def main():
    init_db()
    while True:
        action = input("Do you want to register or login? (register/login/exit): ").strip().lower()

        if action == "register":
            username = input("Enter your username: ").strip()
            register_face(username)
            register_keystroke(username)
        elif action == "login":
            username = input("Enter your username: ").strip()
            if authenticate_face(username) and authenticate_keystroke(username):
                print("Double authentication successful! Access granted.")
            else:
                print("Authentication failed. Access denied.")
        elif action == "exit":
            break
        else:
            print("Invalid option. Please choose 'register', 'login', or 'exit'.")

if __name__ == "__main__":
    main()
