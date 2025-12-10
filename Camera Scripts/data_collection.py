import time
import os
import cv2
import sys
import select
import termios
import tty
from libcamera import Transform
from picamera2 import Picamera2

SAVE_DIR = "dataset"
CLASSES = ["empty", "occupied"]
CAPTURE_INTERVAL = 0.5
IMG_SIZE = 960

for c in CLASSES:
    os.makedirs(os.path.join(SAVE_DIR, c), exist_ok=True)

def get_key_non_blocking():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None

def main():
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        tty.setcbreak(sys.stdin.fileno())

        print("Initializing Picamera2 (Headless Mode)...")
        picam2 = Picamera2()
        
        config = picam2.create_video_configuration(
            main={"size": (1920, 1080), "format": "RGB888"},
            transform=Transform(180),
            controls={"FrameDurationLimits": (100000, 1000000)}
        )
        picam2.configure(config)
        picam2.start()

        print("Applying Configuration Parameters...")
        picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": 500000,
            "ExposureValue": 0.2,
            "NoiseReductionMode": 2,
            "AnalogueGain": 9.85,
            "ColourGains": (1.64, 1.48),
        })
        time.sleep(2)

        print("\n--- HEADLESS DATA COLLECTOR ---")
        print("Controls:")
        print("  [1] : Record EMPTY")
        print("  [2] : Record OCCUPIED")
        print("  [SPACE] : Pause / Stop Recording")
        print("  [q] : Quit")
        print("-------------------------------")

        recording_class = None
        last_capture_time = time.time()
        
        counters = {c: len(os.listdir(os.path.join(SAVE_DIR, c))) for c in CLASSES}

        running = True
        while running:
            frame = picam2.capture_array()
            # Following 6 lines were an attempt to boost the contrast in the training pictures
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            key = get_key_non_blocking()
            
            if key == '1':
                recording_class = "empty"
                print(f"\r>>> RECORDING: EMPTY    ", end='')
            elif key == '2':
                recording_class = "occupied"
                print(f"\r>>> RECORDING: OCCUPIED ", end='')
            elif key == ' ':
                recording_class = None
                print(f"\r>>> PAUSED              ", end='')
            elif key == 'q':
                print("\nQuitting...")
                running = False

            current_time = time.time()
            if recording_class and (current_time - last_capture_time > CAPTURE_INTERVAL):
                small_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                
                filename = f"{int(current_time * 1000)}.jpg"
                save_path = os.path.join(SAVE_DIR, recording_class, filename)
                cv2.imwrite(save_path, small_frame)
                
                counters[recording_class] += 1
                last_capture_time = current_time
                
                print(f"\r>>> REC: {recording_class.upper()} | Count: {counters[recording_class]}   ", end='')

            time.sleep(0.01)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        try:
            picam2.stop()
        except:
            pass

if __name__ == "__main__":
    main()
