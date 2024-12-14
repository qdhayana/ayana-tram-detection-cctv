import cv2
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
from llm_analyze_timestamp_from_image import llm_analyze_timestamp_from_image

def extract_frames(video_path, output_folder, interval=1):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    
    # Print video information
    print(f"FPS: {fps}")
    print(f"Frame Count: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate frames to skip based on interval
    frames_to_skip = fps * interval
    
    current_frame = 0
    frame_number = 0
    frames_with_datetime = 0
    frames_without_datetime = 0
    start_datetime = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Save frame at specified interval
        if current_frame % frames_to_skip == 0:
            # For the first frame, use LLM to analyze timestamp
            if frame_number == 0:
                # Save temporary first frame
                temp_first_frame_path = os.path.join(output_folder, "temp_first_frame.jpg")
                cv2.imwrite(temp_first_frame_path, frame)
                
                # Get timestamp from LLM analysis
                try:
                    datetime_str = llm_analyze_timestamp_from_image(temp_first_frame_path)
                    print(datetime_str)
                    # raise Exception('test')
                    date_str = '2024-12-02 15:07:51'
                    start_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                    print(f"First frame timestamp detected: {start_datetime}")
                    frames_with_datetime += 1
                except Exception as e:
                    print(f"Error analyzing first frame timestamp: {e}")
                    # raise Exception('test')
                    # Fallback to video filename timestamp
                    video_name = os.path.basename(video_path)
                    date_str = video_name.split('_')[3]  # Gets '20241202'
                    time_str = video_name.split('_')[4].split('.')[0]  # Gets '060008'
                    start_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                    frames_without_datetime += 1
                
                # Remove temporary file
                os.remove(temp_first_frame_path)
            
            # Calculate current time by adding seconds based on frame number
            current_time = start_datetime + timedelta(seconds=frame_number)
            
            # Format the filename using the current time
            frame_path = os.path.join(
                output_folder, 
                f"ayana_tram_stop_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            
            cv2.imwrite(frame_path, frame)
            frame_number += 1
            
        current_frame += 1
    
    # Release video capture object
    cap.release()
    print(f"Extracted {frame_number} frames total")
    print(f"Frames with datetime: {frames_with_datetime}")
    print(f"Frames without datetime: {frames_without_datetime}")
    return frame_number

def process_videos(input_dir, base_output_dir, interval=1):
    # Convert input path to Path object
    input_path = Path(input_dir)
    
    # Get all mp4 files
    video_files = list(input_path.glob('*.mp4'))
    
    ## Repair
    # video_files = [v for v in video_files if 'ayana_tram_stop_20241202_185948.mp4' in str(v)]
    
    total_videos = len(video_files)
    
    print(f"Found {total_videos} videos to process")
    
    # Process statistics
    total_frames = 0
    start_time = time.time()
    
    # Process each video
    for idx, video_path in enumerate(video_files, 1):
        # Create output folder for each video
        video_name = video_path.stem  # Get filename without extension
        output_folder = Path(base_output_dir) / video_name
        
        print(f"\nProcessing video {idx}/{total_videos}: {video_path.name}")
        print(f"Saving frames to: {output_folder}")
        
        # Extract frames
        frames = extract_frames(video_path, output_folder, interval)
        total_frames += frames
        
        # Calculate progress and estimated time
        elapsed_time = time.time() - start_time
        avg_time_per_video = elapsed_time / idx
        remaining_videos = total_videos - idx
        estimated_remaining_time = remaining_videos * avg_time_per_video
        
        print(f"Progress: {idx}/{total_videos} videos processed")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds")

    # Print final statistics
    total_time = time.time() - start_time
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per video: {total_time/total_videos:.2f} seconds")

if __name__ == "__main__":
    target_date = input("Enter date (YYYY-MM-DD): ")
    input_directory = f'dataset/converted/{target_date}'
    output_base_directory = f'dataset/captured_frames/{target_date}'
    interval = 1  # Extract frames every 1 second

    # Run the processing
    process_videos(input_directory, output_base_directory, interval)