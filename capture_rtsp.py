import cv2
import time
import os
from datetime import datetime
import numpy as np
import pytz
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

def capture_rtsp_frames(date,
                        cctv_name, 
                        save_directory="captured_frames", 
                        interval=10,
                        resize_width=800,
                        compression_method='png',
                        quality=9,
                        reconnect_delay=5,
                        max_reconnect_attempts=float('inf')):
    
    save_directory = cctv_name + '_captured_frames/' + date
    rtsp_url = cctvs[cctv_name]
    name = cctv_name.replace('_', ' ').title()

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
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

if __name__ == "__main__":
    print("Available cctv_names:")
    for cctv_name in cctvs:
        print(f"- {cctv_name}")

    cctv_name = input('Enter cctv_name: ')
    if cctv_name not in cctvs:
        print(f"Invalid cctv_name: {cctv_name}")
        exit()

    # Get current Singapore time
    date = (datetime.now(pytz.timezone('Asia/Singapore'))).strftime('%Y%m%d')

    # Start capturing frames
    capture_rtsp_frames(
        date,
        cctv_name,
        interval=10,                    # Capture interval in seconds
        resize_width=800,               # Resize width
        compression_method='png',       # Compression method ('jpeg' or 'png')
        quality=9,                      # Compression quality
        reconnect_delay=2,              # Delay between reconnection attempts
        max_reconnect_attempts=100      # Maximum number of reconnection attempts
    )
