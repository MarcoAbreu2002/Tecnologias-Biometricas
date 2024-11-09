from fpdf import FPDF
import face_recognition
import numpy as np
import sqlite3
import os

# Initialize the database path
DB_PATH = 'face_auth.db'

# Database initialization
def initialize_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Faces (
            id INTEGER PRIMARY KEY,
            person_name TEXT,
            image_path TEXT,
            face_embedding BLOB
        )
        ''')
        conn.commit()
        conn.close()
        os.chmod(DB_PATH, 0o600)

# Secure database connection
def get_db_connection():
    return sqlite3.connect(DB_PATH)

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
    return matches, non_matches

# PDF Report Generation
class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Face Authentication Report', new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_authentication_result(self, image_path, matches, non_matches):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, f'Results for Image: {image_path}', new_x='LMARGIN', new_y='NEXT')

        if matches:
            self.set_font('Helvetica', 'B', 9)
            self.cell(0, 10, 'Matches:', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 9)
            for match in matches:
                self._set_status_color(match[3])
                self.cell(0, 8, f'Matched with {match[0]} (Image: {match[1]}, Distance: {match[2]:.4f}, Status: {self._status_text(match[3])})', new_x='LMARGIN', new_y='NEXT')
                self.set_text_color(0, 0, 0)  # Reset text color to black

        if non_matches:
            self.set_font('Helvetica', 'B', 9)
            self.cell(0, 10, 'Non-matches:', new_x='LMARGIN', new_y='NEXT')
            self.set_font('Helvetica', '', 9)
            for non_match in non_matches:
                self._set_status_color(non_match[3])
                self.cell(0, 8, f'Not matched with {non_match[0]} (Image: {non_match[1]}, Distance: {non_match[2]:.4f}, Status: {self._status_text(non_match[3])})', new_x='LMARGIN', new_y='NEXT')
                self.set_text_color(0, 0, 0)  # Reset text color to black
        
        self.ln(10)

    def _set_status_color(self, success):
        # Set color based on success (green for success, red for failure)
        if success:
            self.set_text_color(0, 128, 0)  # Green for success
        else:
            self.set_text_color(255, 0, 0)  # Red for failure

    def _status_text(self, success):
        return "Success" if success else "Failed"

    def add_statistics(self, total_images, total_matches, total_non_matches):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, '--- Statistics ---', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Helvetica', '', 9)
        
        # Correct accuracy calculation based on total matches and non-matches
        total_attempts = total_matches + total_non_matches
        accuracy = (total_matches / total_attempts) * 100 if total_attempts else 0
        
        self.cell(0, 8, f'Total Images Processed: {total_images}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Total Matches: {total_matches}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Total Non-Matches: {total_non_matches}', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 8, f'Accuracy: {accuracy:.2f}%', new_x='LMARGIN', new_y='NEXT')
        self.ln(10)

# Generate PDF report for a list of input images
def generate_pdf_report(input_images):
    pdf = PDFReport()
    pdf.add_page()

    total_matches = 0
    total_non_matches = 0
    total_images = len(input_images)

    for image_path in input_images:
        matches, non_matches = authenticate_face(image_path)
        pdf.add_authentication_result(image_path, matches, non_matches)

        total_matches += len(matches)
        total_non_matches += len(non_matches)

    # Add statistics at the end
    pdf.add_statistics(total_images, total_matches, total_non_matches)

    pdf_file = "face_auth_report.pdf"
    pdf.output(pdf_file)
    print(f"PDF Report generated as {pdf_file}")

# Initialize and set secure permissions for database file if not present
initialize_db()

# Main menu function
def main_menu():
    while True:
        print("\n--- Face Authentication System ---")
        print("1. Add face to the database")
        print("2. Authenticate face from an image")
        print("3. Generate PDF report for multiple images")  # Updated to specify PDF report
        print("4. Exit")
        choice = input("Choose an option: ")
        
        if choice == '1':
            # Add face to the database
            person_name = input("Enter the person's name: ")
            image_path = input("Enter the path to the person's image: ")
            store_face_embedding(person_name, image_path)
        
        elif choice == '2':
            # Authenticate a single face
            image_path = input("Enter the path to the image for authentication: ")
            matches, non_matches = authenticate_face(image_path)
            
            print("\n--- Authentication Results ---")
            if matches:
                print("Matches found:")
                for match in matches:
                    print(f"Matched with {match[0]} (Image: {match[1]}, Distance: {match[2]:.4f})")
            else:
                print("No matches found.")
        
        elif choice == '3':
            # Generate a PDF report for multiple images
            num_images = int(input("Enter the number of images to authenticate: "))
            input_images = []
            for i in range(num_images):
                image_path = input(f"Enter the path for image {i+1}: ")
                input_images.append(image_path)
            
            # Call the updated PDF generation function
            generate_pdf_report(input_images)  # Updated function name
        
        elif choice == '4':
            # Exit the program
            print("Exiting program.")
            break
        
        else:
            print("Invalid choice. Please try again.")


# Run the main menu
main_menu()

