from auth import store_face_embedding, authenticate_face
from report import generate_pdf_report
from utils import log_event, calculate_accuracy

def main_menu():
    while True:
        print("\n--- Face Authentication System ---")
        print("1. Add face to the database")
        print("2. Authenticate face from an image")
        print("3. Generate PDF report for multiple images")
        print("4. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            person_name = input("Enter the person's name: ")
            image_path = input("Enter the path to the person's image: ")
            result, time_taken = store_face_embedding(person_name, image_path)
            print(f"Embedding stored in {time_taken:.4f} seconds.")
        
        elif choice == '2':
            image_path = input("Enter the path to the image for authentication: ")
            (matches, non_matches), time_taken = authenticate_face(image_path)
            print(f"Authentication completed in {time_taken:.4f} seconds.")
            print("\n--- Authentication Results ---")
            if matches:
                print("Matches found:")
                for match in matches:
                    print(f"Matched with {match[0]} (Image: {match[1]}, Distance: {match[2]:.4f})")
            else:
                print("No matches found.")
        
        elif choice == '3':
            num_images = int(input("Enter the number of images to authenticate: "))
            input_images = [input(f"Enter the path for image {i+1}: ") for i in range(num_images)]
            generate_pdf_report(input_images)
        
        elif choice == '4':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
