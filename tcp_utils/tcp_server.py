import socket, time

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('localhost', 1234)
print ('starting up on {} port {}'.format(server_address[0], server_address[1]))
sock.bind(server_address)
# Listen for incoming connections
sock.listen(1)

# Wait for a connection
print ('waiting for a connection')
connection, client_address = sock.accept()
try:
    print ('connection from: ', client_address)

    # Receive the data in small chunks and retransmit it
    for i in range(0, 30):
        connection.sendall(bytes("Test data: {}".format(i), 'ascii'))
        time.sleep(1)

finally:
    # Clean up the connection
    connection.close()