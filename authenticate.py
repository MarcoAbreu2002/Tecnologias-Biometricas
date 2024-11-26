import os
import time
import sqlite3
import numpy as np
from pynput import keyboard
import cv2
import face_recognition
from colorama import Fore, Style, init
import sys
import time

# Initialize colorama for colored text
init(autoreset=True)

# Database settings
DB_NAME = "multi_auth.db"
SENTENCE = "The quick brown fox jumps over the lazy dog."

# Clear the screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def clear_input():
    sys.stdout.write("\033[K")  
    sys.stdout.flush()

# Initialize the database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id_code TEXT PRIMARY KEY,
            face_embedding BLOB,
            avg_dwell_time REAL,
            avg_flight_time REAL
        )
    ''')
    conn.commit()
    conn.close()

# Register face
def register_face(id_code):
    print(f"{Fore.CYAN}Please face the camera to register your face.")
    cap = cv2.VideoCapture(0)
    registered = False

    while not registered:
        ret, frame = cap.read()
        if not ret:
            print(f"{Fore.RED}Error accessing the webcam. Please try again.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 1:
            face_embedding = face_encodings[0].tobytes()

            # Save to database
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO users (id_code, face_embedding) VALUES (?, ?)',
                    (id_code, sqlite3.Binary(face_embedding))
                )
                conn.commit()
                conn.close()
                registered = True
                print(f"{Fore.GREEN}Face registered successfully!")
            except sqlite3.IntegrityError:
                print(f"{Fore.RED}ID code already exists. Please use a unique ID code.")
                break
            except Exception as e:
                print(f"{Fore.RED}Error inserting into database: {e}")
                break
        else:
            print(f"{Fore.YELLOW}Ensure only your face is visible and retry.")

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Register keystroke dynamics
def register_keystroke(id_code):
    print(f"{Fore.CYAN}Please type the following sentence exactly as shown:")
    print(f"{Fore.YELLOW}{SENTENCE}")
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

        if len(key_times) > 1:
            last_two_keys = list(key_times.keys())[-2:]
            if all(k in key_times for k in last_two_keys):
                flight_time = key_times[last_two_keys[1]] - key_times[last_two_keys[0]]
                flight_times.append(flight_time)

        if key == keyboard.Key.enter:
            input()
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
            UPDATE users 
            SET avg_dwell_time = ?, avg_flight_time = ? 
            WHERE id_code = ?
        ''', (avg_dwell_time, avg_flight_time, id_code))
        conn.commit()
        conn.close()

        print(f"{Fore.GREEN}Keystroke data registered successfully!")
    else:
        print(f"{Fore.RED}Not enough keystrokes to register. Please try again.")

# Authenticate face
def authenticate_face():
    print(f"{Fore.CYAN}Authenticating face... Please ensure your face is clearly visible.")
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    authenticated = False

    while not authenticated:
        elapsed_time = time.time() - start_time
        if elapsed_time > 15:
            print(f"{Fore.RED}Authentication timed out. No face recognized within 15 seconds.")
            cap.release()
            cv2.destroyAllWindows()
            return None

        ret, frame = cap.read()
        if not ret:
            print(f"{Fore.RED}Error accessing the webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) == 1:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('SELECT id_code, face_embedding FROM users')
            users = cursor.fetchall()
            conn.close()

            for id_code, stored_embedding_blob in users:
                stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float64)
                distance = np.linalg.norm(stored_embedding - face_encodings[0])
                if distance < 0.6:
                    authenticated = True
                    print(f"{Fore.GREEN}Welcome user {id_code}! Proceed with keystroke authentication.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return id_code
        else:
            print(f"{Fore.YELLOW}Face not recognized. Please adjust your position.")

        cv2.imshow("Authenticate Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"{Fore.RED}Authentication failed. Please try again.")
    return None

# Authenticate keystroke dynamics
def authenticate_keystroke(id_code):
    print(f"{Fore.CYAN}Please type the following sentence exactly as shown:")
    print(f"{Fore.YELLOW}{SENTENCE}")
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

        if len(key_times) > 1:
            last_two_keys = list(key_times.keys())[-2:]
            if all(k in key_times for k in last_two_keys):
                flight_time = key_times[last_two_keys[1]] - key_times[last_two_keys[0]]
                flight_times.append(flight_time)

        if key == keyboard.Key.enter:
            input()
            return False   

    key_times = {}
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    if len(dwell_times) > 1:
        avg_dwell_time = np.mean(dwell_times)
        avg_flight_time = np.mean(flight_times)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT avg_dwell_time, avg_flight_time FROM users WHERE id_code = ?', (id_code,))
        result = cursor.fetchone()
        conn.close()

        if result:
            stored_dwell_time, stored_flight_time = result
            if abs(avg_dwell_time - stored_dwell_time) < 0.05 and abs(avg_flight_time - stored_flight_time) < 0.05:
                print(f"{Fore.GREEN}Keystroke authentication successful!")
                return True
            else:
                print(f"{Fore.RED}Typing pattern does not match. Authentication failed.")
                return False
    print(f"{Fore.RED}Authentication failed.")
    return False

# About the project
def about_project():
    print(f"{Fore.MAGENTA}\nAbout the Project")
    print(f"{Style.BRIGHT}This is a multi-factor authentication system that uses:")
    print(f"1. {Fore.YELLOW}Face recognition{Fore.RESET} for initial authentication.")
    print(f"2. {Fore.YELLOW}Keystroke dynamics{Fore.RESET} for behavioral verification.")
    print(f"\nThe project is designed to enhance security and user experience.\n")

# Main menu
def main():
    init_db()
    clear_screen()
    while True:
        print(f"{Fore.MAGENTA}Welcome to the Multi-Factor Authentication System!")
        print(f"{Style.BRIGHT}Secure your access with Face Recognition and Keystroke Dynamics.")
        print("-" * 50)

        print(f"{Fore.BLUE}Menu:")
        print(f"1 - Register")
        print(f"2 - Login")
        print(f"3 - About the Project")
        print(f"4 - Exit")
        action = input(f"{Fore.CYAN}Choose an option (1/2/3/4): ").strip()

        if action == "1":
            id_code = input(f"{Fore.CYAN}Enter your unique identification code: ").strip()
            if not id_code.isdigit():
                print(f"{Fore.RED}Identification code must be numeric. Please try again.")
                input(f"{Fore.YELLOW}Press Enter to continue...")
                continue
            register_face(id_code)
            register_keystroke(id_code)
            input(f"{Fore.YELLOW}Press Enter to continue...")
            clear_screen()
        elif action == "2":
            id_code = authenticate_face()
            if id_code and authenticate_keystroke(id_code):
                print(f"{Fore.GREEN}Double authentication successful! Access granted.")
            else:
                print(f"{Fore.RED}Authentication failed. Access denied.")
            input(f"{Fore.YELLOW}Press Enter to continue...")
            clear_screen()
        elif action == "3":
            about_project()
            input(f"{Fore.YELLOW}Press Enter to return to the main menu...")
            clear_screen()
        elif action == "4":
            print(f"{Fore.MAGENTA}Exiting the system. Goodbye!")
            break
        else:
            print(f"{Fore.RED}Invalid option. Please choose 1, 2, 3, or 4.")
            input(f"{Fore.YELLOW}Press Enter to continue...")

if __name__ == "__main__":
    main()
