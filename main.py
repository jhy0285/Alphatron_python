import cv2
import websocket

cap = cv2.VideoCapture(0)

ws = websocket.WebSocket()
ws.connect("ws://192.168.0.23:7777/test")
print("WebSocket connected")

def send_frame_via_websocket(frame):
    try:
        frame = cv2.resize(frame, (640, 480))
        encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1].tobytes()
        frame_size = len(encoded_frame).to_bytes(4, byteorder='big')
        ws.send_binary(frame_size + encoded_frame)
        print("Frame sent successfully")
    except Exception as e:
        print(f"Error sending frame: {e}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    send_frame_via_websocket(frame)
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ws.close()
print("WebSocket closed")
