import socket
import json

HOST = "127.0.0.1"  # Connect to the local TCP server
PORT = 5006        # Must match the server port

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

while True:
    data = client.recv(1024)  # Receive data
    time_info = json.loads(data.decode('utf-8'))  # Parse JSON data
    print(f"Received time data: {time_info}")  # Display received data
