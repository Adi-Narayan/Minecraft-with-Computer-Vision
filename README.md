# Pose-Controlled Input System

## Overview
This is a personal project that uses computer vision and pose estimation to control computer inputs (keyboard and mouse) based on body movements. The system leverages the MediaPipe Pose solution to detect human landmarks and translates specific gestures into actions such as clicking, jumping, or moving the mouse cursor.

## Features
- **Squat Detection**: Squatting triggers the 'Ctrl' key press.
- **Jump Gesture**: Raising both hands triggers the 'Space' key press.
- **Mouse Control**: 
  - Prayer pose (hands close together) moves the mouse cursor up or down.
  - Elbow positioning controls left or right mouse movement.
- **Click Actions**: Specific arm angles trigger left or right mouse clicks.
- **Walk/Run**: Leg movements control the 'W' key for walking or a combination of 'W', 'Shift', and 'Space' for running/jumping.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- PyDirectInput (`pydirectinput`)
- NumPy (`numpy`)
- A webcam for real-time pose detection

## Installation
1. Clone or download this repository.
2. Install the required packages:
   ```bash
   pip install opencv-python mediapipe pydirectinput numpy
   ```
3. Ensure you have a working webcam connected.

## Usage
1. Run the main script:
   ```bash
   python main.py
   ```
2. Position yourself in front of the webcam.
3. Perform gestures to control inputs:
   - I have interchanged 'Ctrl' and 'Shift' in my game for my accessibility, # MAKE SURE YOU CHANGE THIS PART IN YOUR GAME OR IN THE CODE!
   - Squat to press 'Ctrl'.
   - Raise both hands to jump ('Space').
   - Bring hands close together and move them up/down for mouse cursor movement.
   - Adjust arm angles for left/right clicks.
   - Move legs to simulate walking or running.

5. Press `Esc` to exit the program.


## Notes
- The script assumes additional Python scripts (`mouse_up.py`, `mouse_down.py`, `mouse_left.py`, `mouse_right.py`) are present in the specified directory for mouse movement. Ensure these are available or modify the paths accordingly.
- The system is designed for real-time webcam input and does not process static images by default (though the code includes commented-out static image processing).
- Adjust the landmark thresholds (e.g., `hand_distance`, `normalfactor`) based on your setup for better accuracy.

## Limitations
- Requires a well-lit environment for accurate pose detection.
- Gesture detection may vary based on body proportions and camera quality.
- The mouse movement scripts are Windows-specific due to hardcoded file paths.

## License
This is a personal project and not licensed for commercial use. Feel free to modify and experiment with the code for personal learning purposes.
