import os
import subprocess
import datetime
import json
from tqdm import tqdm

def convert_video(input_file, output_file):
    """Convert a single video file using FFmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libx264',
            '-preset', 'medium',    # medium preset
            '-crf', '17',          # high quality
            '-y',                  # overwrite output file if exists
            output_file
        ]
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, result.stderr
        return True, None
        
    except Exception as e:
        return False, str(e)

def save_progress(progress_file, completed_files, failed_files):
    """Save progress to a JSON file"""
    with open(progress_file, 'w') as f:
        json.dump({
            'completed': completed_files,
            'failed': failed_files
        }, f)

def load_progress(progress_file):
    """Load progress from a JSON file"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return data.get('completed', []), data.get('failed', [])
    return [], []

def convert_videos_sequential(input_dir, output_dir):
    """Convert all AVI videos one by one with resume capability"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Progress file path
    progress_file = os.path.join(output_dir, 'conversion_progress.json')
    
    # Load previous progress if exists
    completed_files, failed_files = load_progress(progress_file)
    
    # Get all .avi files
    video_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.avi'):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_filename = os.path.splitext(file)[0] + '.mp4'
                output_path = os.path.join(output_subdir, output_filename)
                
                # Skip if already completed
                if input_path not in completed_files:
                    video_files.append((input_path, output_path))
    
    if not video_files:
        print("No new videos to convert!")
        return

    total_videos = len(video_files)
    print(f"Found {total_videos} videos to convert")
    if completed_files:
        print(f"Resuming from previous session ({len(completed_files)} already completed)")
    
    # Calculate total size of remaining videos
    total_size_mb = sum(os.path.getsize(input_file) for input_file, _ in video_files) / (1024 * 1024)
    print(f"Total size of remaining videos: {total_size_mb:.2f} MB")
    print("\nConversion settings:")
    print("- Preset: medium")
    print("- CRF: 17 (high quality)")
    
    # Start conversion
    start_time = datetime.datetime.now()
    
    try:
        # Process each video with progress bar
        for i, (input_file, output_file) in enumerate(video_files, 1):
            # Get video file size
            file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
            
            print(f"\nProcessing {i}/{total_videos}: {os.path.basename(input_file)}")
            print(f"File size: {file_size_mb:.2f} MB")
            
            # Record start time for this video
            video_start_time = datetime.datetime.now()
            
            # Convert video
            success, error = convert_video(input_file, output_file)
            
            # Calculate time taken for this video
            video_time = datetime.datetime.now() - video_start_time
            
            if success:
                output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"✓ Success! Time taken: {video_time}")
                print(f"Output size: {output_size_mb:.2f} MB")
                print(f"Size ratio: {(output_size_mb/file_size_mb):.2f}x")
                completed_files.append(input_file)
            else:
                print(f"✗ Failed! Time taken: {video_time}")
                print(f"Error: {error}")
                failed_files.append((input_file, error))
            
            # Save progress after each video
            save_progress(progress_file, completed_files, failed_files)
            
            # Calculate and show overall progress
            elapsed_total = datetime.datetime.now() - start_time
            avg_time_per_video = elapsed_total / i
            videos_remaining = total_videos - i
            estimated_remaining = avg_time_per_video * videos_remaining
            
            print(f"\nOverall Progress: {i}/{total_videos} ({(i/total_videos)*100:.1f}%)")
            print(f"Total time elapsed: {elapsed_total}")
            print(f"Estimated time remaining: {estimated_remaining}")
            print("-" * 80)
    
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user!")
        print("Progress has been saved. You can resume later by running the script again.")
        return
    
    # Print final summary
    print("\nConversion Complete!")
    print(f"Total time taken: {datetime.datetime.now() - start_time}")
    print(f"Successfully converted: {len(completed_files)}")
    print(f"Total failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed conversions:")
        for file, error in failed_files:
            print(f"- {os.path.basename(file)}: {error}")

if __name__ == "__main__":
    # Set your paths here
    date = input("Enter date (YYYY-MM-DD): ")
    input_directory = f"dataset/original/{date}"
    output_directory = f"dataset/converted/{date}"
    
    # Start conversion
    convert_videos_sequential(input_directory, output_directory)