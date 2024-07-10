# Road Lane-Line Detection Project

## Brief Description

This project involves developing an AI-powered application to accurately detect road lane lines from video footage. Utilizing advanced computer vision techniques and machine learning algorithms, the application processes video frames to identify and highlight lane markings, ensuring improved safety and navigation for autonomous driving systems.

## Features

- Real-time road lane detection
- Video processing with lane line overlay
- User-friendly interface for uploading and viewing videos
- Streamlit-based web application for easy deployment and access

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Road Lane-Line Detection Project is designed to enhance the safety and navigation capabilities of autonomous vehicles by accurately identifying lane markings on the road. This application processes video footage to detect and highlight lane lines, providing real-time feedback to the user.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Murasajo/Road-Lane-Line-Detection.git
    cd road-lane-detection
    ```

2. **Create a Virtual Environment**:

    ```bash
    python -m venv myenv
    source myenv/bin/activate   # On Windows, use `myenv\Scripts\activate`
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit Application**:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the Streamlit application in your browser.
2. Upload a video file using the file uploader interface.
3. The application will process the video and display both the original and lane-detected video side by side.

## Project Structure

```
road-lane-detection/
├── app.py                  # Main application script for Streamlit
├── gui.py                  # GUI script for local video display
├── requirements.txt        # List of project dependencies
├── road_clip.mp4           # Sample input video file
├── road_video.mp4          # Sample input video file
└── README.md               # Project documentation
```

## Dependencies

The project relies on the following main packages:

- streamlit
- opencv-python-headless
- numpy
- pillow
- moviepy
- matplotlib

For a complete list of dependencies, refer to the `requirements.txt` file.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please open an issue or create a pull request.
