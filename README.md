# Face-Controlled Mouse System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision-based mouse control system that enables hands-free computer interaction using facial gestures. Control your mouse cursor by moving your head, and perform clicks by opening your mouth or blinking.

**Author:** Song Tan  
**Affiliation:** School of Information Science and Engineering, Ningbo University

---

## Features

- **Head Tracking Control** - Move your head to control the mouse cursor position
- **Mouth Gesture Click** - Open your mouth to trigger left-click
- **Blink Gesture Click** - Blink to trigger right-click
- **Real-time Visual Feedback** - On-screen display of face landmarks and cursor position
- **Multi-level Smoothing** - Exponential smoothing and moving average filtering for stable control
- **Automatic Model Download** - Downloads required MediaPipe model on first run
- **Configurable Parameters** - Adjustable sensitivity, smoothing, and detection thresholds

---

## Demo

```
┌─────────────────────────────────────────────────────────────┐
│  Face Controlled Mouse                    FPS: 30.0         │
│  Screen: (1920, 540)                                        │
│                                                             │
│     Head Move = Mouse Move                                  │
│     Open Mouth = Left Click                                 │
│     Blink = Right Click                                     │
│     'c' Calibrate | 'q' Quit                                │
│                                                             │
│                        ┌─────────┐                          │
│                        │  Face   │                          │
│                        │   ●     │  ← Nose tracking point   │
│                        │  /|\    │                          │
│                        └─────────┘                          │
│                                                             │
│                                         +                   │
│                                         │ ← Virtual cursor  │
│                                       ──┼──                 │
│                                         │                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.7 or higher
- A webcam (built-in or external)
- Camera access permissions

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python mediapipe numpy pyautogui
```

---

## Quick Start

### Basic Usage

```bash
cd face_mouse_controller
python main.py
```

### Command Line Options

```bash
python main.py --help

# Available options:
#   -c, --camera ID        Camera device ID (default: 0)
#   -w, --width WIDTH      Camera width resolution (default: 320)
#   -H, --height HEIGHT    Camera height resolution (default: 240)
#   -s, --sensitivity R    Mouse sensitivity 0.1-1.0 (default: 0.2)
#   --smooth FACTOR        Smoothing factor 0-1 (default: 0.5)
#   --dead-zone PIXELS     Dead zone threshold (default: 20)
#   --mouth-threshold PX   Mouth open threshold (default: 15)
```

### Examples

```bash
# Use external camera (ID 1) with higher sensitivity
python main.py -c 1 -s 0.3

# Lower resolution for better performance
python main.py -w 160 -H 120

# Smoother but slower response
python main.py --smooth 0.7 --dead-zone 30
```

---

## Controls

| Gesture | Action |
|---------|--------|
| Move Head | Move mouse cursor |
| Open Mouth | Left click |
| Blink | Right click |
| Press `c` | Calibrate/Reset |
| Press `q` | Quit application |
| Press `s` | Save screenshot |

---

## Project Structure

```
face_mouse_controller/
├── main.py                 # Entry point with CLI
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration and parameters
│   ├── filters.py         # Smoothing algorithms
│   ├── face_detector.py   # Face detection module
│   ├── mouse_controller.py # Mouse control module
│   └── application.py     # Main application logic
└── face_landmarker.task   # MediaPipe model (auto-downloaded)
```

---

## Architecture

### Module Overview

| Module | Description |
|--------|-------------|
| `Config` | Centralized configuration management with dataclass |
| `KalmanFilter` | State estimation for noise reduction |
| `MovingAverageFilter` | Temporal smoothing filter |
| `ExponentialSmoother` | Responsive yet smooth output |
| `CompositeFilter` | Combined filtering pipeline |
| `FaceDetector` | MediaPipe FaceLandmarker wrapper |
| `MouseController` | Screen mapping and mouse operations |
| `Visualizer` | OpenCV overlay rendering |
| `FaceMouseApplication` | Main application orchestrator |

### Data Flow

```
Camera Frame
     │
     ▼
┌─────────────┐
│ FaceDetector│ ──── Detect face landmarks
└─────────────┘
     │
     ▼
┌─────────────┐
│ HeadPosition│ ──── Calculate head position
└─────────────┘
     │
     ▼
┌─────────────┐
│   Filters   │ ──── Apply smoothing
└─────────────┘
     │
     ▼
┌─────────────┐
│  Screen Map │ ──── Map to screen coordinates
└─────────────┘
     │
     ▼
┌─────────────┐
│   Mouse     │ ──── Move cursor / Click
└─────────────┘
```

---

## Configuration

Key parameters can be adjusted in `src/config.py` or via command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_WIDTH` | 320 | Camera resolution width |
| `CAMERA_HEIGHT` | 240 | Camera resolution height |
| `EFFECTIVE_RATIO` | 0.2 | Head movement to screen ratio |
| `SMOOTH_FACTOR` | 0.5 | Exponential smoothing (0-1) |
| `DEAD_ZONE` | 20 | Minimum movement threshold (px) |
| `MOUTH_OPEN_THRESHOLD` | 15 | Mouth open detection threshold |
| `JITTER_THRESHOLD` | 5 | Click stability threshold |

---

## Troubleshooting

### Camera Not Opening

1. Ensure camera is not used by another application
2. Grant camera access in System Preferences → Security & Privacy
3. Try a different camera ID: `python main.py -c 1`

### Model Download Failed

The MediaPipe model is downloaded automatically. If it fails:

1. Check your internet connection
2. Download manually from: [MediaPipe Face Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
3. Place the file in the project directory

### Poor Tracking Performance

1. Ensure good lighting conditions
2. Keep your face centered in the camera view
3. Avoid face occlusions (masks, hands, etc.)
4. Adjust `--sensitivity` for your preferred responsiveness

---

## Technical Details

### Dependencies

- **OpenCV** (`cv2`) - Camera capture and image processing
- **MediaPipe** - Face landmark detection
- **PyAutoGUI** - Mouse control operations
- **NumPy** - Numerical computations

### Smoothing Algorithm

The system uses a multi-stage smoothing pipeline:

1. **Dead Zone Filter** - Ignores micro-movements below threshold
2. **Exponential Smoothing** - Blends current position with history
3. **Moving Average** - Averages recent positions for stability

### Performance Optimization

- Reduced camera resolution (320x240) for faster processing
- Single face detection mode
- Disabled unused MediaPipe outputs
- Efficient NumPy operations

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for face landmark detection
- [OpenCV](https://opencv.org/) for computer vision utilities

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{face_mouse_controller,
  author = {Tan, Song},
  title = {Face-Controlled Mouse System},
  year = {2024},
  institution = {Ningbo University, School of Information Science and Engineering}
}
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

