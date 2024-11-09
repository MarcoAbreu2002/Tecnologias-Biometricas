import numpy as np
import face_recognition
from database import get_db_connection
from utils import log_event, time_execution

@time_execution
def store_face_embedding(person_name, image_path):
    """Store face embedding in the database with time logging."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 1:
        face_embedding = encodings[0]
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Faces (person_name, image_path, face_embedding) VALUES (?, ?, ?)",
                       (person_name, image_path, face_embedding.tobytes()))
        conn.commit()
        conn.close()
        log_event(f"Stored embedding for {person_name} from {image_path}")
    else:
        log_event(f"Error: Found {len(encodings)} faces in {image_path}, expected exactly one.")


# Function to authenticate face and retrieve matches and non-matches
def authenticate_face(image_path, threshold=0.6):
    conn = get_db_connection()
    cursor = conn.cursor()

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) != 1:
        print(f"Error: Found {len(encodings)} faces in {image_path}, expected exactly one.")
        conn.close()
        return [], []
    
    target_encoding = encodings[0]
    cursor.execute("SELECT person_name, image_path, face_embedding FROM Faces")
    results = cursor.fetchall()

    matches = []
    non_matches = []

    for person_name, db_image_path, db_embedding_blob in results:
        db_embedding = np.frombuffer(db_embedding_blob, dtype=np.float64)
        distance = np.linalg.norm(target_encoding - db_embedding)

        if distance < threshold:
            matches.append((person_name, db_image_path, distance, True))  # Adding True for successful match
        else:
            non_matches.append((person_name, db_image_path, distance, False))  # Adding False for non-match

    conn.close()
    print(matches)
    print(non_matches)
    return matches, non_matches


