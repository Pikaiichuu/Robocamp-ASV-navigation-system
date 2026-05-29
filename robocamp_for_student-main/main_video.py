import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# ---------- Konfigurasi ----------
MODEL_PATH = os.path.join("..", "model_asv_2024", "bola.pt")
VIDEO_PATH = "Video_kapal.mp4"
CONF_THRESH = 0.35
PIXEL_TO_METER = 0.01  # Asumsi: 1 pixel = 0.01 meter
STRAIGHT_ZONE_PX = 60

# Class ID
CLASS_BUOY_GREEN = 0  # Class ID untuk bola hijau
CLASS_BUOY_RED  = 1  # Class ID untuk bola biru

# Warna dalam format BGR (Blue, Green, Red)
COLOR_GREEN = (0, 255, 0)   # Hijau
COLOR_RED  = (0, 0, 255)   # Merah
COLOR_MIDPOINT = (255, 255, 0)  # Cyan

midpoint_history = deque(maxlen=8)  # Menyimpan posisi midpoint terakhir

def main():
    # Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model tidak ditemukan: {MODEL_PATH}")
        return
        
    model = YOLO(MODEL_PATH)
    
    # Buka Video
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video tidak ditemukan: {VIDEO_PATH}")
        return
        
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Tidak dapat membuka video.")
        return

    print("Memproses video... Tekan 'q' untuk keluar.")

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Video selesai.")
            break

        frame_h, frame_w = frame.shape[:2]
        frame_cx = frame_w // 2

        green_centers = []
        red_centers = []

        # Inference YOLO pada frame
        results = model(frame, conf=CONF_THRESH, stream=True, verbose=False, imgsz=640)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Klasifikasi berdasarkan class ID
                # Class 0: Green (Hijau), Class 1: Red (direpresentasikan sebagai Biru)
                if cls_id == 0:
                    color = COLOR_GREEN
                    label = f"Bola Hijau {conf:.0%}"
                    green_centers.append((cx, cy))
                elif cls_id == 1:
                    color = COLOR_RED
                    label = f"Bola Merah {conf:.0%}"
                    red_centers.append((cx, cy))
                else:
                    continue

                # Gambar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Gambar label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Navigasi
            nav_command = "Cari_Buoy"
            offset_px = 0
            offset_m = 0.0
            midpoint_x = None

            if green_centers and red_centers:
                best_green = max(green_centers, key=lambda c: c[1])
                best_red = max(red_centers, key=lambda c: c[1])

                cv2.line(frame, best_green, best_red, (255, 255, 0), 2)

                raw_mid = (best_green[0] + best_red[0]) // 2
                midpoint_history.append(raw_mid)
                midpoint_x = int(np.mean(midpoint_history))

                offset_px = midpoint_x - frame_cx
                offset_m = offset_px * PIXEL_TO_METER

                if abs(offset_px) < STRAIGHT_ZONE_PX:
                    nav_command = "Lurus"
                elif offset_px < 0:
                    nav_command = "Kanan"
                else:
                    nav_command = "Kiri"

            elif green_centers:
                nav_command = "Kanan"
            elif red_centers:
                nav_command = "Kiri"

            # Overlay informasi navigasi
            if midpoint_x is not None:
                mid_y = frame_h // 2
                cv2.circle(frame, (midpoint_x, mid_y), 10, (255, 255, 0), -1)
                cv2.putText(frame, "Midpoint", (midpoint_x + 14, mid_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Panel Info
            fps = 1.0 / (time.time() - frame_start + 1e-9)
            panel = [
                (f"Arah : {nav_command}", (0, 255, 255)),
                (f"Buoy Merah : {len(red_centers)}", COLOR_RED),
                (f"Buoy Hijau : {len(green_centers)}", COLOR_GREEN),
                (f"Obstacle : 0", (255, 255, 255)),
                (f"FPS : {fps:.1f}", (255, 255, 255)),
                (f"=> Koreksi {nav_command} Offset: {offset_m:.2f} m", (0, 200, 255))
            ]

            cv2.rectangle(frame, (5, 5), (320, 175), (40, 40, 40), -1)
            for i, (text, col) in enumerate(panel):
                cv2.putText(frame, text, (12, 30 + i * 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                
            banner = f"=> ARAHKAN KAPAL KE {nav_command}  {offset_m:.2f} m"
            cv2.rectangle(frame, (0, frame_h - 60), (frame_w, frame_h), (30, 30, 30), -1)
            cv2.putText(frame, banner, (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Offset: {offset_px:+d}px", (20, frame_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Tampilkan frame
        cv2.imshow("Deteksi Bola Hijau & Biru - Video", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")

if __name__ == "__main__":
    main()
