import cv2
import os
import pytz
from datetime import datetime

def get_singapore_time():
    """Get current time in Singapore timezone"""
    return datetime.now(pytz.timezone('Asia/Singapore'))

def should_stop_capture():
    """Check if it's time to stop capturing (11 PM Singapore time)"""
    return get_singapore_time().hour >= 23

def compress_frame(frame, method='png', quality=9):
    """Compress frame using different methods"""
    encode_param = (
        [int(cv2.IMWRITE_JPEG_QUALITY), quality] if method == 'jpeg' 
        else [int(cv2.IMWRITE_PNG_COMPRESSION), quality//10]
    )
    ext = '.jpg' if method == 'jpeg' else '.png'
    _, buffer = cv2.imencode(ext, frame, encode_param)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

def connect_rtsp(rtsp_url, max_retries=3, retry_delay=2):
    """Attempt to connect to RTSP stream with retries"""
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            print(f"Successfully connected to RTSP stream on attempt {attempt + 1}")
            return cap
        print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    return None

def create_directories(base_path, cctv_name, date):
    """Create and return required directories"""
    directories = {
        'save': f'{base_path}/{cctv_name}/{date}/captured_frames',
        'detection': f'{base_path}/{cctv_name}/{date}/output',
        'processed': f'{base_path}/{cctv_name}/{date}/processed_frames'
    }
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    return directories

def process_detection_results(results, frame, timestamp, filename, directories):
    """Process and save detection results"""
    if len(results[0].boxes) > 0:
        annotated_frame = results[0].plot()
        height, _ = annotated_frame.shape[:2]

        cv2.putText(
            annotated_frame,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            (10, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (0, 255, 0),
            2
        )
        
        output_path = os.path.join(directories['detection'], f"detected_{filename}")
        cv2.imwrite(output_path, annotated_frame)
        
        print("Object detected:")
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                print(f"- {class_name}: {confidence:.2f}")
    else:
        print("No object detected.")