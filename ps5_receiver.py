import os
import socket
import time

FIFO = "/tmp/ps5-joystick"
PORT = 50100

if not os.path.exists(FIFO):
    os.mkfifo(FIFO)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(0.5)

print(f"PS5-Empfänger wartet auf UDP-Port {PORT}")

fd = None
last_packet = time.monotonic()

try:
    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            if time.monotonic() - last_packet > 1:
                if fd is not None:
                    os.close(fd)
                    fd = None
            continue

        last_packet = time.monotonic()

        if not data or len(data) % 8:
            continue

        if fd is None:
            try:
                fd = os.open(
                    FIFO, os.O_WRONLY | os.O_NONBLOCK
                )
            except OSError:
                continue

        try:
            os.write(fd, data)
        except OSError:
            os.close(fd)
            fd = None

except KeyboardInterrupt:
    pass
finally:
    if fd is not None:
        os.close(fd)
    sock.close()
