# AYANA Tram Detection System

## Overview
This project implements a custom object detection system for AYANA's trams using YOLO11. The system captures frames from an RTSP stream and processes them to detect and monitor AYANA's trams.

## Features
- RTSP stream frame capture
- Automatic reconnection for unstable RTSP streams
- Automatic capture termination at 11 PM (Singapore time)
- PNG compression with optimal quality
- Build custom YOLO11 model for AYANA's tram detection
- Detect AYANA's tram based on captured images

