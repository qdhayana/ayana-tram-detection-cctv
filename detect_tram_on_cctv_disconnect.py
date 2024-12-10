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
            return
        
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
            
            height, _ = annotated_frame.shape[:2]

            # Add timestamp to image
            cv2.putText(
                annotated_frame,
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                (10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5,
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

        # Move processed image
        shutil.move(image_path, os.path.join(processed_directory, filename))
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

def detect_tram_on_cctv(cctv_name, 
                        interval=10,
                        resize_width=800,
                        compression_method='png',
                        quality=9,
                        reconnect_delay=10,
                        max_reconnect_attempts=float('inf'),
                        model=None,
                        confidence_threshold=0.5,
                        display_width=640,
                        display_feed=True):  # Add display_feed parameter

    # Get current Singapore time
    date = (datetime.now(pytz.timezone('Asia/Singapore'))).strftime('%Y%m%d')

    # Validate CCTV name
    if cctv_name not in cctvs:
        print("Invalid CCTV name: `{}`!\nPlease use available options:\n{}".format(
            cctv_name, '\n'.join(["- `{}`".format(cctv) for cctv in cctvs.keys()])))
        exit()

    rtsp_url = cctvs[cctv_name]
    name = cctv_name.replace('_', ' ').title()
    window_name = f'CCTV Stream - {name}'

    # ... (directory creation code remains the same)

    try:
        while True:
            if should_stop_capture():
                print("\nStopping capture - reached 11 PM Singapore time")
                break

            try:
                # Connect to RTSP stream
                print(f"{name}: Connecting to RTSP stream...")
                cap = connect_rtsp(rtsp_url)
                
                if cap is None:
                    # Close any existing windows when connection fails
                    if display_feed:
                        cv2.destroyAllWindows()
                    reconnect_count += 1
                    if reconnect_count >= max_reconnect_attempts:
                        print(f"{name}: Exceeded maximum reconnection attempts ({max_reconnect_attempts})")
                        return
                    print(f"{name}: Could not establish connection. Attempt {reconnect_count} of {max_reconnect_attempts}")
                    print(f"Waiting {reconnect_delay} seconds before retry...")
                    time.sleep(reconnect_delay)
                    continue

                # Reset reconnect count on successful connection
                reconnect_count = 0

                # Capture one frame
                ret, frame = cap.read()
                
                if not ret:
                    # Close window if frame capture fails
                    if display_feed:
                        cv2.destroyAllWindows()
                    reconnect_count += 1
                    if reconnect_count >= max_reconnect_attempts:
                        print(f"{name}: Exceeded maximum reconnection attempts ({max_reconnect_attempts})")
                        return
                    print(f"{name}: Failed to capture frame. Attempt {reconnect_count} of {max_reconnect_attempts}")
                    print(f"Will retry in {reconnect_delay} seconds...")
                    cap.release()
                    time.sleep(reconnect_delay)
                    continue

                # Process the captured frame
                try:
                    # Resize frame
                    height, width = frame.shape[:2]
                    aspect_ratio = width / height
                    new_width = resize_width
                    new_height = int(new_width / aspect_ratio)
                    resized_frame = cv2.resize(frame, (new_width, new_height), 
                                             interpolation=cv2.INTER_AREA)

                    # Compress frame
                    compressed_frame = compress_frame(resized_frame, 
                                                   method=compression_method,
                                                   quality=quality)

                    # Save frame with timestamp
                    sg_time = get_singapore_time()
                    timestamp = sg_time.strftime("%Y%m%d_%H%M%S")
                    ext = '.jpg' if compression_method == 'jpeg' else '.png'
                    filename = f"frame_{timestamp}{ext}"
                    filepath = os.path.join(save_directory, filename)
                    
                    cv2.imwrite(filepath, compressed_frame)

                    # Display frame if display_feed is True
                    if display_feed:
                        display_height = int(display_width * (height/width))
                        display_frame = cv2.resize(frame, (display_width, display_height))
                        cv2.imshow(window_name, display_frame)

                    # Get file size and print info
                    file_size = os.path.getsize(filepath) / 1024
                    print(f"{name}: saved frame - {filename} (Size: {file_size:.1f} KB)")

                    # Perform object detection
                    detecting_object(model, confidence_threshold, save_directory, 
                                   filename, detection_directory, processed_directory)

                except Exception as e:
                    print(f"Error processing frame: {str(e)}")

                finally:
                    # Release the capture object
                    cap.release()
                    print(f"{name}: Disconnected from RTSP stream")
                    # Close the window when disconnected if display_feed is True
                    if display_feed:
                        cv2.destroyWindow(window_name)

                # Wait for the specified interval before next capture
                print(f"Waiting {interval} seconds before next capture...")
                for i in range(interval):
                    if display_feed:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\nUser requested to stop...")
                            return
                    time.sleep(1)

            except KeyboardInterrupt:
                print("\nStopping capture...")
                break
                
            except Exception as e:
                # Close window on unexpected errors
                if display_feed:
                    cv2.destroyAllWindows()
                reconnect_count += 1
                if reconnect_count >= max_reconnect_attempts:
                    print(f"{name}: Exceeded maximum reconnection attempts ({max_reconnect_attempts})")
                    return
                print(f"Unexpected error: {str(e)}")
                print(f"Attempt {reconnect_count} of {max_reconnect_attempts}")
                print(f"Waiting {reconnect_delay} seconds before retry...")
                time.sleep(reconnect_delay)

    finally:
        # Ensure everything is properly cleaned up
        if 'cap' in locals() and cap is not None:
            cap.release()
        if display_feed:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-cctv_name', '--cctv_name', required=True,
                    help='Name of the CCTV camera to capture from')
    ap.add_argument('-display', '--display_feed', 
                    action='store_true',
                    help='Display live feed (default: False)')
    args = vars(ap.parse_args())

    # Load the trained model
    model_path = "build_custom_model/runs/train/20241209_231950/weights/best.pt"
    model = YOLO(model_path)

    print(f"Model loaded from: {model_path}")
    detect_tram_on_cctv(args['cctv_name'], 
                        interval=10,
                        resize_width=800,
                        compression_method='png',
                        quality=9,
                        reconnect_delay=5,
                        max_reconnect_attempts=float('inf'),
                        model=model,
                        display_width=640,
                        display_feed=args['display_feed'])  # Pass the display_feed argument
