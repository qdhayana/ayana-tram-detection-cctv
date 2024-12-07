from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import pytz
from rtsp_conf import cctvs

def detect_tram(
    date,
    cctv_name,
    model_path,  # Replace with your actual model path
    confidence_threshold=0.25
):
    """
    Detect trams in images using the trained model
    """
    image_directory = 'production/' + cctv_name + '_captured_frames/' + date
    output_directory = 'production/' + cctv_name + '_detected_results/' + date

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load the trained model
    model = YOLO(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Get Singapore timezone
    sg_tz = pytz.timezone('Asia/Singapore')
    
    # Process each image
    for filename in sorted(os.listdir(image_directory)):
        if filename.endswith(('.jpg', '.png')):
            try:
                # Full path to image
                image_path = os.path.join(image_directory, filename)
                
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image: {filename}")
                    continue
                
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
                    
                    # Get confidence scores
                    confidences = [float(box.conf) for box in results[0].boxes]
                    max_conf = max(confidences)
                    
                    print(f"Tram detected in {filename} with confidence: {max_conf:.2f}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

if __name__ == "__main__":
    # Get current Singapore time
    # date = (datetime.now(pytz.timezone('Asia/Singapore'))).strftime('%Y%m%d')

    print("Available cctv_names:")
    for cctv_name in cctvs:
        print(f"- {cctv_name}")

    cctv_name = input('Enter cctv_name: ')
    if cctv_name not in cctvs:
        print(f"Invalid cctv_name: {cctv_name}")
        exit()

    # If screenshots data are not available, stop the process
    if os.path.exists('production/' + cctv_name + '_captured_frames/') == False:
        print('No screenshots data available. Please run capture_rtsp.py first.')
        exit()

    date = input('Enter date (YYYYMMDD): ')

    # Replace with your actual model path
    model_path = "build_custom_model/runs/train/20241207_082659/weights/best.pt"  # Update this path if needed

    # Run detection
    detect_tram(
        date,
        cctv_name,
        model_path,
        confidence_threshold=0.5
    )