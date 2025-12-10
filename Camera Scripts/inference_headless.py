import time
import cv2
import numpy as np
import tensorflow.lite as tflite
from libcamera import Transform
from picamera2 import Picamera2

# --- CONFIGURATION ---
MODEL_PATH = "room_detector_960.tflite"
IMG_SIZE = 960
THRESHOLD = 0.5
CLASS_NAMES = ["EMPTY", "OCCUPIED"]

def main():
    print("Loading TFLite Model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    print(f"Model Input Shape: {input_shape}")

    print("Initializing Camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1920, 1080), "format": "RGB888"},
        transform=Transform(180),
        controls={"FrameDurationLimits": (100000, 1000000)}
    )
    picam2.configure(config)
    picam2.start()

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

    print("\n--- ROOM MONITOR (TFLITE) RUNNING ---")
    print("Press CTRL+C to Quit")

    try:
        while True:
            start_time = time.time()

            frame = picam2.capture_array()
            small_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            input_data = small_frame.astype('float32') / 255.0
            
            input_data = np.expand_dims(input_data, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction_score = output_data[0][0]

            if prediction_score > THRESHOLD:
                result = CLASS_NAMES[1]
                confidence = prediction_score
            else:
                result = CLASS_NAMES[0]
                confidence = 1.0 - prediction_score

            fps = 1.0 / (time.time() - start_time)

            print(f"\r>>> STATUS: [{result}] ({confidence*100:.1f}%) | Speed: {fps:.1f} FPS    ", end='')

    except KeyboardInterrupt:
        print("\nStopping...")
        picam2.stop()

if __name__ == "__main__":
    main()
