## Before and After Processing

<div style="display: flex; justify-content: space-around;">
    <div style="text-align: center;">
        <p><strong>Before Processing</strong></p>
        <img src="images/before_processing.png" alt="Before Processing" width="45%">
    </div>
    <div style="text-align: center;">
        <p><strong>After Processing</strong></p>
        <img src="images/after_processing.png" alt="After Processing" width="45%">
    </div>
</div>

# Facial-Emotional Detection for Videos using AWS Rekognition

This project utilizes **AWS Rekognition** to detect and analyze emotions from faces in video files. The detected emotions are overlaid onto the video frames, and the processed video is saved with emotion-based highlights. 

## Features

- **Emotion Detection:** Detects emotions such as *Calm*, *Confused*, *Sad*, *Disgusted*, *Angry*, *Surprised*, *Happy*, and *Fear* in faces.
- **Video Processing:** Overlays detected emotions on video frames and outputs a processed video with visual indicators of emotions.
- **AWS Integration:** Leverages AWS Rekognition for accurate emotion detection.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python 3.11**
- **AWS Account** with credentials for Rekognition service.
- **S3 Bucket** to store the input video file.

### Installation

Install the required Python packages:

```bash
pip install boto3 opencv-python numpy
```

### Configuration

Before running the script, update the following placeholders in the code:

- **AWS Credentials**: Replace `aws_access_key_id`, `aws_secret_access_key`, and `region_name` with your AWS credentials and region.

    ```python
    client = boto3.client('rekognition',
                          aws_access_key_id='<Your-Access-Key-ID>',
                          aws_secret_access_key='<Your-Secret-Access-Key>',
                          region_name='<Your-Region>')
    ```

- **S3 Bucket and Video Name**: Replace `Bucket` and `Name` with your S3 bucket name and video file name.

    ```python
    response = client.start_face_detection(
        Video={'S3Object': {'Bucket': '<Your-S3-Bucket>',
                            'Name': '<Your-Video-Name>'}},
        FaceAttributes='ALL'
    )
    ```

- **Input and Output Paths**: Specify the local paths for the input video file and the output video file.

    ```python
    input_path = '<Your-Input-Video-Path>'
    output_path = '<Your-Output-Video-Path>'
    ```

### Usage

1. Upload the input video file to your S3 bucket.
2. Run the script to start detecting emotions and processing the video:

    ```bash
    python facial_emotion_detection.py
    ```

3. The output video with emotion highlights will be saved at the specified `output_path`.

### Example

- **Input Video:** `female elderly home 5.mp4` (stored in S3 bucket)
- **Output Video:** `output_video.mp4` (saved locally)

## Emotion Colors

The following colors are used to represent emotions:

- **CALM**: Green
- **CONFUSED**: Blue
- **SAD**: Magenta
- **DISGUSTED**: Cyan
- **ANGRY**: Red
- **SURPRISED**: Yellow
- **HAPPY**: Dark Green
- **FEAR**: Purple

## Acknowledgments

- **AWS Rekognition** for providing powerful facial analysis capabilities.
- **OpenCV** for video processing.
