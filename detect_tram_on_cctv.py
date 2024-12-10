import cv2
import argparse
import time
import os
from datetime import datetime
import numpy as np
import pytz
from ultralytics import YOLO
import shutil
from rtsp_conf import cctvs

def get_singapore_time():
    """Get current time in Singapore timezone"""
    sg_tz = pytz.timezone('Asia/Singapore')
    sg_time = datetime.now(sg_tz)
    return sg_time

def should_stop_capture():
    """Check if it's time to stop capturing (11 PM Singapore time)"""
    sg_time = get_singapore_time()
    return sg_time.hour >= 23  # 23:00 = 11 PM

def compress_frame(frame, method='png', quality=9):
    """Compress frame using different methods"""
    if method == 'jpeg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    elif method == 'png':
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), quality//10]
        _, buffer = cv2.imencode('.png', frame, encode_param)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    return frame

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

def detecting_object(model, confidence_threshold, image_directory, filename, output_directory, processed_directory):
    # Get Singapore timezone
    sg_tz = pytz.timezone('Asia/Singapore')

    try:
        # Full path to image
        image_path = os.path.join(image_directory, filename)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {filename}")
        
        # Get timestamp from filename (assuming format "frame_YYYYMMDD_HHMMSS.jpg")
        timestamp_str = filename.split('frame_')[1].split('.')[0]
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        timestamp = sg_tz.localize(timestamp)
        
        # Perform detection
        results = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False
        )
        
        # Process results
        if len(results[0].boxes) > 0:
            # Get the image for drawing
            annotated_frame = results[0].plot()
            
            # Add timestamp to image
            cv2.putText(
                annotated_frame,
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Save annotated image
            output_path = os.path.join(output_directory, f"detected_{filename}")
            cv2.imwrite(output_path, annotated_frame)
            
            # Print detections in a compact format
            print("Object detected:")
            for result in results:
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    confidence = float(box.conf)
                    print(f"- {class_name}: {confidence:.2f}")
            
        else:
            print(f"No object detected.")

        shutil.move(image_path, processed_directory)
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        
def detect_tram_on_cctv(cctv_name, 
                        interval=10,
                        resize_width=800,
                        compression_method='png',
                        quality=9,
                        reconnect_delay=5,
                        max_reconnect_attempts=float('inf'),
                        model=None,
                        confidence_threshold=0.5,):
    
    # Get current Singapore time
    date = (datetime.now(pytz.timezone('Asia/Singapore'))).strftime('%Y%m%d')

    # Check if cctv_name is valid
    valid_cctv = list(cctvs.keys())
    valid_cctv = ["- `{}`".format(cctv) for cctv in valid_cctv]
    if cctv_name not in cctvs:
        print("Invalid CCTV name: `{}`!\nPlease use available options:\n{}".format(cctv_name, '\n'.join(valid_cctv)))
        exit()

    rtsp_url = cctvs[cctv_name]
    name = cctv_name.replace('_', ' ').title()

    # Create save directory
    save_directory = 'production/' + cctv_name + '/' + date + '/captured_frames'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create detection directory
    detection_directory = 'production/' + cctv_name + '/' + date + '/output'
    if not os.path.exists(detection_directory):
        os.makedirs(detection_directory)

    # Create processed directory
    processed_directory = 'production/' + cctv_name + '/' + date + '/processed_frames/'
    if not os.path.exists(processed_directory):
        os.makedirs(processed_directory)
    
    reconnect_count = 0
    last_capture_time = 0
    
    while True:
        if should_stop_capture():
            print("\nStopping capture - reached 11 PM Singapore time")
            break

        try:
            cap = connect_rtsp(rtsp_url)
            if cap is None:
                print(f"{name}: Could not establish initial connection. Retrying...")
                time.sleep(reconnect_delay)
                continue

            print(f"{name}: Connected to RTSP stream. Capturing frame every {interval} seconds...")
            sg_time = get_singapore_time()
            print(f"Current Singapore time: {sg_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            while True:
                if should_stop_capture():
                    print("\nStopping capture - reached 11 PM Singapore time")
                    return

                ret, frame = cap.read()
                
                if not ret:
                    print("Lost connection to stream. Attempting to reconnect...")
                    reconnect_count += 1
                    
                    if reconnect_count > max_reconnect_attempts:
                        print(f"Exceeded maximum reconnection attempts ({max_reconnect_attempts})")
                        return
                    
                    cap.release()
                    cap = connect_rtsp(rtsp_url)
                    if cap is None:
                        print(f"Reconnection failed. Waiting {reconnect_delay} seconds before next attempt...")
                        time.sleep(reconnect_delay)
                        continue
                    
                    print(f"{name}: Successfully reconnected to stream")
                    continue
                
                reconnect_count = 0
                
                # Display the frame
                cv2.imshow(f'CCTV Stream - {name}', frame)
                
                # Check if it's time to capture a frame
                current_time = time.time()
                if current_time - last_capture_time >= interval:
                    try:
                        # Calculate new dimensions maintaining aspect ratio
                        height, width = frame.shape[:2]
                        aspect_ratio = width / height
                        new_width = resize_width
                        new_height = int(new_width / aspect_ratio)
                        
                        # Resize the frame
                        resized_frame = cv2.resize(frame, (new_width, new_height), 
                                             interpolation=cv2.INTER_AREA)
                        
                        # Compress the frame
                        compressed_frame = compress_frame(resized_frame, 
                                                       method=compression_method,
                                                       quality=quality)
                        
                        # Generate timestamp for filename (in Singapore time)
                        sg_time = get_singapore_time()
                        timestamp = sg_time.strftime("%Y%m%d_%H%M%S")
                        ext = '.jpg' if compression_method == 'jpeg' else '.png'
                        filename = f"frame_{timestamp}{ext}"
                        filepath = os.path.join(save_directory, filename)
                        
                        # Save frame
                        cv2.imwrite(filepath, compressed_frame)
                        
                        # Get file size
                        file_size = os.path.getsize(filepath) / 1024  # Size in KB
                        print(f"{name}: saved frame - {filename} (Size: {file_size:.1f} KB)")
                        
                        last_capture_time = current_time

                        # Detect AYANA's Tram
                        detecting_object(model, confidence_threshold, save_directory, filename, detection_directory, processed_directory)
                        
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
                        continue
                
                # Check for 'q' key to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested to stop...")
                    return
                
        except KeyboardInterrupt:
            print("\nStopping capture...")
            break
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print("Attempting to restart capture process...")
            time.sleep(reconnect_delay)
            continue
            
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-cctv_name', '--cctv_name', required=True)
    args = vars(ap.parse_args())

    # Load the trained model
    ## Replace with your actual model path
    model_path = "build_custom_model/runs/train/20241209_231950/weights/best.pt"  # Update this path if needed
    model = YOLO(model_path)

    print(f"Model loaded from: {model_path}")
    detect_tram_on_cctv(args['cctv_name'], 
                        interval=10,
                        resize_width=800,
                        compression_method='png',
                        quality=9,
                        reconnect_delay=5,
                        max_reconnect_attempts=float('inf'),
                        model=model)