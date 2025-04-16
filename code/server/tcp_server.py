import socket
import json
import time

HOST = "0.0.0.0"
PORT = 5006         # Choose an available port

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(5)

print(f"TCP Server started on {HOST}:{PORT}")

while True:
    client, addr = server.accept()
    print(f"Connected to {addr}")

    try:
        while True:
            # Send current time
            curr_time = time.time()
            tag_info = {"curr time": curr_time}
            print(f"Sending time: {curr_time}")
            message = json.dumps(tag_info) + "\n" 
            client.sendall(message.encode('utf-8'))

            time.sleep(1)  # Prevent sending too fast
            # tag_info = {"x": 1.2, "y": 3.4, "theta": 0.5}
            # message = json.dumps(tag_info) + "\n"  # Append newline
            # client.send(message.encode('utf-8'))
    except (BrokenPipeError, ConnectionResetError):
        print(f"Client {addr} disconnected")
        client.close()
