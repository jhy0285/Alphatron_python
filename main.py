import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import DeepOCSORT
from pathlib import Path
import websocket
import json
import struct

# YOLOv8n 모델 로드
model = YOLO('yolov8n.pt')

# DeepOCSORT 트래커 초기화
weights_path = Path(r"osnet_x0_25_msmt17.pt")  # 모델 가중치 경로 설정
tracking_config = 'DeepOCSORT/config.yaml'  # 트래킹 설정 파일 경로 설정

tracking_method = DeepOCSORT(
    model_weights=weights_path,
    device='cpu',
    fp16=False,
    config_file=tracking_config
)

def send_centroids(ws, centroids):
    """스프링 서버에 중점 좌표 전송"""
    try:
        if centroids:
            # 중점 좌표 리스트를 문자열로 변환하여 전송 (ID, x, y 형태)
            message = ','.join([f'{item["id"]},{item["cx"]},{item["cy"]}' for item in centroids])
            ws.send(message.encode())
            print(f"Sent centroids: {message}")
        else:
            message = "None,None,None"
            ws.send(message.encode())
    except Exception as e:
        print(f"Error sending centroids: {e}")

def process_frame(frame):
    """
    프레임을 처리하여 객체 탐지 및 추적을 수행하고, 중점 좌표를 반환합니다.
    """
    # YOLOv8n 모델로 객체 탐지 수행
    results = model(frame, classes=[0], verbose=False)  # 클래스 0(person)만 탐지
    detections = results[0].boxes.data.cpu().numpy()

    # DeepOCSORT로 객체 추적
    tracks = tracking_method.update(detections, frame)

    centroids = []  # 중점 좌표 리스트

    # 추적된 객체 그리기 및 중점 계산
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id, conf = map(int, track[:7])
        cx = (x1 + x2) // 2  # 중점 x 좌표
        cy = (y1 + y2) // 2  # 중점 y 좌표
        centroids.append({'id': track_id, 'cx': cx, 'cy': cy})  # ID 포함

        # 바운딩 박스 및 ID 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
    return frame, centroids

def process_and_send_frame(frame, ws):
    """
    프레임을 처리하고, 결과를 WebSocket으로 전송합니다.
    """
    # 프레임 처리
    processed_frame, centroids = process_frame(frame)

    # WebSocket으로 데이터 전송
    try:
        centroids_json = json.dumps(centroids)
        centroids_bytes = centroids_json.encode('utf-8')
        centroid_length = len(centroids_bytes)

        # 프레임을 JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_data = buffer.tobytes()
        frame_length = len(frame_data)

        # 데이터 패킹
        message = struct.pack('>I', centroid_length) + centroids_bytes + struct.pack('>I', frame_length) + frame_data

        # WebSocket으로 데이터 전송
        ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)
        print("Sent frame and centroids over websocket")

    except Exception as e:
        print(f"Error sending data over websocket: {e}")

    return processed_frame, centroids

def capture_and_process(ws):
    """카메라에서 영상을 캡처하여 처리하고 서버로 전송"""
    cap = cv2.VideoCapture(0)  # 카메라 장치 번호 0 사용

    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        # 프레임 처리 및 WebSocket으로 전송
        processed_frame, centroids = process_and_send_frame(frame, ws)

        # 이미지 출력
        cv2.imshow('YOLO + DeepOCSORT Tracking', processed_frame)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # WebSocket 연결 설정
    try:
        ws = websocket.create_connection("ws://localhost:7777/python", ping_interval=None)
        print("WebSocket client connected to ws://localhost:7777/python")
    except Exception as e:
        print(f"Error connecting to WebSocket server: {e}")
        return

    try:
        capture_and_process(ws)
    except KeyboardInterrupt:
        print("Capture process interrupted.")
    finally:
        ws.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
