# AYANA Tram Detection System

## Overview
- This project implements a custom object detection system for AYANA's trams using YOLO11. The system captures frames from an RTSP stream and processes them to detect and monitor AYANA's trams.
- A comprehensive system for analyzing tram arrivals using video processing and computer vision. This project processes surveillance footage to detect and track trams, calculating their arrival and departure times at designated stops.

## Features
- RTSP stream frame capture
- Automatic reconnection for unstable RTSP streams
- Automatic capture termination at 11 PM (Singapore time)
- PNG compression with optimal quality
- Build custom YOLO11 model for AYANA's tram detection
- Detect AYANA's tram based on captured images

## Update 2024-12-11
We updated the flow because we do not have a suitable device for smooth live-streaming. 
1. Obtain the exported CCTV video from the Security Department (*.avi).
2. Convert the *.avi files to *.mp4. We need to convert the files due to issues with video codecs.
3. Extract frames from each *.mp4 file at 1-second intervals.
4. Detect the tram movements.

## Project Flow

The system operates in five sequential steps:

1. Video Conversion
2. Frame Extraction
3. Timestamp Analysis
4. Area Definition
5. Tram Detection and Analysis

## Prerequisites

- Python 3.8+
- FFmpeg
- OpenCV
- YOLO model (trained for tram detection)
- OpenAI API key
- Required Python packages:
  ```bash
  pip install opencv-python numpy pandas tqdm ultralytics langchain-openai shapely
  ```

## Project Structure
```
├── build_custom_model/                  # Codes to create the custom YOLO11 model
├── dataset/                            
│   └── captured_frames/
│       └── YYYY-MM-DD/                  # Extracted frame images
│   └── converted/
│       └── YYYY-MM-DD/                  # Converted MP4 videos
│   └── original/
│       └── YYYY-MM-DD/                  # Source AVI videos
│   └── debug_frames/
│       └── YYYY-MM-DD/                  # Debugging detection results
├── demo/                                # Example frame images to test the tram arrival detection
├── live-feed/                           # Implementation on RTSP
├── area_coordinates.txt                 # Generated from `determine_area_coordinates.py`
├── capturing_frames.py
├── convert_avi_to_mp4.py
├── determine_area_coordinates.py
├── llm_analyze_timestamp_from_image.py
├── rename_avi.ipynb
└── tram_arrival_detection.ipynb
```

## Detailed Process Description

### 1. Video Conversion (`convert_avi_to_mp4.py`)
Converts surveillance footage from AVI to MP4 format with optimized settings.

**Process Details:**
- Uses FFmpeg for high-quality video conversion
- Maintains original video quality with x264 codec
- Features:
  - Resume capability for interrupted conversions
  - Progress tracking with ETA
  - Maintains folder structure
  - JSON-based progress logging
  - Error handling with detailed logs
  - Size and quality ratio reporting

**Settings:**
- Preset: medium (balanced speed/quality)
- CRF: 17 (high quality)
- Codec: libx264

**Usage:**
```bash
python convert_avi_to_mp4.py
# Enter date when prompted (YYYY-MM-DD)
```

### 2. Frame Extraction (`capturing_frames.py`)
Extracts frames at regular intervals from the converted MP4 videos.

**Process Details:**
- Extracts one frame per second (configurable)
- Maintains temporal sequence
- Features:
  - Automatic timestamp extraction
  - Progress tracking with ETA
  - Organized output structure
  - Memory-efficient processing
  - FPS-aware extraction
  - Multi-video batch processing

**Output Format:**
- Frame filename: `ayana_tram_stop_YYYYMMDD_HHMMSS.jpg`
- Organized in date-based folders

**Usage:**
```bash
python capturing_frames.py
# Enter date when prompted (YYYY-MM-DD)
```

### 3. Timestamp Analysis (`llm_analyze_timestamp_from_image.py`)
Uses GPT-4 Vision to extract timestamps from video frames.

**Process Details:**
- Analyzes timestamps in frame images
- Uses LangChain for API interaction
- Features:
  - Automatic retry mechanism
  - Error handling
  - Custom prompt configuration
  - Base64 image encoding
  - Fallback timestamp extraction from filename
  - Configurable API settings

**Analysis Process:**
1. Encodes image to base64
2. Sends to GPT-4 Vision API
3. Processes response to extract timestamp
4. Validates timestamp format
5. Provides fallback mechanisms

### 4. Area Definition (`determine_area_coordinates.py`)
Interactive tool for defining the tram detection area using OpenCV.

**Process Details:**
- Creates interactive window for area selection
- Supports 8-point polygon definition
- Features:
  - Real-time visual feedback
  - Point-by-point drawing
  - Line connection visualization
  - Reset capability
  - Coordinate saving
  - Interrupt handling

**User Interface:**
- Left-click: Place points
- 'r': Reset selection
- 'c': Confirm selection
- 'q': Quit without saving

**Usage:**
```bash
python determine_area_coordinates.py
```

### 5. Tram Detection (`tram_arrival_detection.ipynb`)
Analyzes frames to detect and track trams using YOLO model.

**Process Details:**
- Uses custom-trained YOLO model
- Tracks tram presence in defined area
- Features:
  - Real-time detection visualization
  - Duration calculation
  - Detailed reporting
  - CSV export
  - Debug frame generation
  - Progress tracking
  - Comprehensive logging

**Analysis Steps:**
1. Loads YOLO model and area coordinates
2. Processes frames sequentially
3. Detects trams using YOLO
4. Checks tram position relative to defined area
5. Calculates entry/exit times
6. Generates reports and visualizations

**Output Files:**
1. Debug Frames:
   - Shows detection boundaries
   - Visualizes area definition
   - Indicates tram presence/absence
   - Includes timestamps

2. CSV Report (`tram_arrivals.csv`):
   - Arrival number
   - Entry time
   - Exit time
   - Duration in seconds

3. Analysis Report (`analysis_report.txt`):
   - Total frames analyzed
   - Arrival summary
   - Detailed timeline
   - Statistical analysis

## Data Flow
```
[AVI Videos] → [MP4 Videos] → [Extracted Frames] → [Timestamp Analysis] → [Area Definition] → [Tram Detection] → [Analysis Reports]
```

## Performance Considerations

### Video Conversion
- CPU-intensive process
- Requires sufficient disk space
- Progress is saved for large batches

### Frame Extraction
- Memory-efficient processing
- Configurable extraction rate
- Handles large video files

### Timestamp Analysis
- API rate limits consideration
- Retry mechanism for reliability
- Fallback systems in place

### Area Definition
- One-time setup process
- Coordinates are reusable
- Lightweight operation

### Tram Detection
- GPU recommended for YOLO
- Batch processing capable
- Memory usage scales with frame count

## Important Notes

1. Ensure proper folder structure before running scripts
2. Configure OpenAI API key in `llm_analyze_timestamp_from_image.py`
3. Run scripts in the specified order
4. Keep backup of source videos
5. Monitor disk space for extracted frames

## Error Handling

- Each script includes error handling and logging
- Progress is saved regularly during video conversion
- Interrupted processes can be resumed
- Check log outputs for any issues

## License

[Specify your license here]

## Contact

[Your contact information]