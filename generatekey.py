from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
import os

# Function to generate AES encryption key
def generate_aes_key():
    # AES key size can be 128, 192, or 256 bits (16, 24, or 32 bytes)
    key_size = 32  # 256-bit AES key
    key = os.urandom(key_size)
    return key

# Function to save the key to a file
def save_key_to_file(key, file_name):
    with open(file_name, 'wb') as file:
        file.write(key)
    print(f"AES key has been saved to {file_name}")

# Main program execution
def main():
    key = generate_aes_key()
    save_key_to_file(key, 'aes_key.key')

if __name__ == "__main__":
    main()
