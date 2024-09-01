import boto3
import cv2
import time
import json
import numpy as np

# Define color mapping for emotions
emotion_colors = {
    'CALM': (0, 255, 0),  # Green
    'CONFUSED': (255, 0, 0),  # Blue
    'SAD': (255, 0, 255),  # Magenta
    'DISGUSTED': (0, 255, 255),  # Cyan
    'ANGRY': (0, 0, 255),  # Red
    'SURPRISED': (255, 255, 0),  # Yellow
    'HAPPY': (0, 128, 0),  # Dark Green
    'FEAR': (128, 0, 128)  # Purple
}

# Initialize a session using Amazon Rekognition
client = boto3.client('rekognition',     
                      aws_access_key_id= '',# e.g. AKIA4MTWHJJEEV2CJ37G
                      aws_secret_access_key='', # e.g. RRchVhvSnpEMpcaJwzmgYqvbLuwaqXazl6o5+OoH
                      region_name='') # e.g. us-east-2

# Start face detection by aws rekonition
response = client.start_face_detection(
    Video={'S3Object': {'Bucket': '', # e.g. facial-emotional-analysis
                        'Name': ''}}, # e.g. female elderly home 5.mp4
    FaceAttributes='ALL'
)

job_id = response['JobId']
print(f"Job started with ID: {job_id}")

# Wait for the aws rekonition to complete 
while True:
    response = client.get_face_detection(JobId=job_id)
    status = response['JobStatus']
    print(f"Job status: {status}")
    # Check job status
    if status in ['SUCCEEDED', 'FAILED']:
        break
    
    time.sleep(5)  # Wait before checking the status again

# "response" contains a list of dictionaries, each dictionary represents a detected face with detailed info
response = client.get_face_detection(JobId=job_id)
# Extract and print the results in JSON format
results = json.dumps(response, indent=4)
# print(results)
# Optionally, save the results to a JSON file
with open('face_detection_results.json', 'w') as f:
    f.write(results)
# Extract video metadata and frame rate
video_metadata = response.get('VideoMetadata', {})

frame_rate = video_metadata.get('FrameRate', 30.0)  # Default to 30.0 if not found

# Get face detection results from aws rekonition (for each timeframe)
face_detection_results = response['Faces'] # fps = 2 
# In python , a generator is an iterator that yields item one at a time 

# Extract all timestamps from face_detection_results
all_timestamps = [face['Timestamp'] for face in face_detection_results]


# Path to the input video and output videoface_detection_results
input_path = "" # e.g. C:\\Users\\user\\Downloads\\female elderly home 5.mp4
output_path = '' # e.g. C:\\Users\\user\\Desktop\\aws_rekonition\\output_video.mp4

# Create a VideoCapture object and read from the input file
cap = cv2.VideoCapture(input_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'X264' for H.264
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

# Define the order of emotions
emotion_order = ["CALM", "HAPPY", "SURPRISED", "ANGRY", "DISGUSTED", "FEAR", "CONFUSED", "SAD"]
# frame rate per second
frame_count = 0
# Initialize a variable to store the last detected emotions
last_detected_emotions = None

while cap.isOpened():  # cap_count =  16 (length of video in second) x 30 (frame per second)  = 485

    # cap.read() : reads the next frame from the input video file
    # ret : true/false on whether the frame was successfully read
    # frame : the actual frame data read from the video (3-d Array)
    ret, frame = cap.read()
    if ret == False:
        break

    # Calculate the timestamp for the current frame in milliseconds
    frame_timestamp = int((frame_count / frame_rate) * 1000)

    # Find the nearest detection timestamp
    nearest_timestamp = max([t for t in all_timestamps if t <= frame_timestamp ], default=0)
    difference = frame_timestamp - nearest_timestamp
    if difference > 550:  
        out.write(frame) # Write the unmodified frame to the output video
        frame_count += 1
        continue
    
    # print("current_capture_frame_timestamp:", frame_timestamp, "nearest_timestamp: ", nearest_timestamp) # finds the element in detection_timestamps for which the function lambda x: abs(x - frame_timestamp) returns the smallest value.


    # Find all faces in the current frame, based on the nearest timestamp
    current_faces = [face for face in face_detection_results if face['Timestamp'] == nearest_timestamp]


    if current_faces: # If there is face data analysed from aws in this timeframe
        emotions = current_faces[0]['Face']['Emotions'] # curren_faces[0] is to only extract the first person face data
        dominant_emotion = emotions[0]["Type"]
        confidence_score = emotions[0]["Confidence"]
        last_detected_emotions = dominant_emotion  # update last_detected_emotions
    else: # for the time before any face data is detected by aws
        emotions = None
        dominant_emotion = None



    # Set the frame color based on the dominant emotion
    if dominant_emotion in emotion_colors:
        frame_color = emotion_colors[dominant_emotion]
    else:
        frame_color = (0, 0, 0)  # Default to black if emotion is not recognized


    if emotions:
        # Overlay all emotions and their confidence scores on the frame
        y_position = 50  # Initial y position for the text
        bar_start_x = 200  # Starting x position for the bar chart
        bar_max_length = 300  # Maximum length of the bar representing 100% confidence

        # Create a dictionary for easy lookup
        emotion_dict = {emotion['Type']: emotion['Confidence'] for emotion in emotions}

        for emotion_type in emotion_order:
            if emotion_type in emotion_dict:
                confidence = emotion_dict[emotion_type]
                color = emotion_colors.get(emotion_type, (255, 255, 255))  # Default to white if emotion not in mapping

                # Draw the emotion type
                cv2.putText(frame, f"{emotion_type}", (50, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw the bar representing the confidence
                bar_length = int((confidence / 100) * bar_max_length)
                cv2.rectangle(frame, (bar_start_x, y_position - 10), (bar_start_x + bar_length, y_position + 10), color, -1)

                y_position += 30  # Move to the next line for the next emotion

        for face in current_faces: # accept multiple faces within a frame
            box = face['Face']['BoundingBox']
            left = int(box['Left'] * frame_width)
            top = int(box['Top'] * frame_height)
            width = int(box['Width'] * frame_width)
            height = int(box['Height'] * frame_height)
            box_color = emotion_colors.get(dominant_emotion, (255, 255, 255))  # Use dominant emotion color for bounding box
            # Draw the bounding box
            cv2.rectangle(frame, (left, top), (left + width, top + height), box_color, 2)

            # Overlay the dominant emotion text
            if dominant_emotion:
                emotion_text = f"{dominant_emotion} {confidence_score:.2f}%"
                # Put text above the box
                cv2.putText(frame, emotion_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Write the modified frame to the output video
    out.write(frame)
    frame_count += 1

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video processing complete")