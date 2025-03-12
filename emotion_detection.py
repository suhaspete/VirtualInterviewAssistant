import cv2
import numpy as np
import streamlit as st
import time
import random

class EmotionDetector:
    """A class for detecting emotions from facial expressions using OpenCV"""
    
    def __init__(self):
        """Initialize the emotion detector"""
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion labels
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'confident']
        
        # Initialize trackers for motion and position
        self.prev_face_position = None
        self.motion_history = []
        self.last_position_time = time.time()
        self.stability_counter = 0
        self.last_emotion = None
        self.emotion_stability = 0
    
    def detect_faces(self, frame):
        """Detect faces in the image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces, gray
    
    def analyze_emotion(self, frame):
        """
        Detect faces and analyze emotion.
        
        This is a simplified emotion detection that uses face position, size,
        and movement patterns as proxies for emotions.
        """
        faces, gray = self.detect_faces(frame)
        
        if len(faces) == 0:
            # If no face is detected, return the last emotion if it exists
            return self.last_emotion, frame
        
        # Get the largest face (closest to camera)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Calculate face size relative to frame as a confidence proxy
        frame_area = frame.shape[0] * frame.shape[1]
        face_area = w * h
        face_ratio = face_area / frame_area
        
        # Position metrics
        center_x = x + w/2
        center_y = y + h/2
        frame_center_x = frame.shape[1]/2
        frame_center_y = frame.shape[0]/2
        
        distance_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
        normalized_distance = distance_from_center / (np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)/2)
        
        # Movement tracking
        current_time = time.time()
        current_position = (center_x, center_y)
        
        movement_amount = 0
        if self.prev_face_position is not None:
            # Calculate movement since last frame
            movement = np.sqrt((current_position[0] - self.prev_face_position[0])**2 + 
                             (current_position[1] - self.prev_face_position[1])**2)
            time_diff = current_time - self.last_position_time
            if time_diff > 0:
                movement_amount = movement / time_diff
                self.motion_history.append(movement_amount)
                # Keep only recent motion history
                if len(self.motion_history) > 10:
                    self.motion_history.pop(0)
        
        self.prev_face_position = current_position
        self.last_position_time = current_time
        
        # Calculate average motion if we have history
        avg_motion = sum(self.motion_history) / max(len(self.motion_history), 1)
        
        # Determine face vertical position (looking up/down)
        vertical_position = center_y / frame.shape[0]  # 0 at top, 1 at bottom
        
        # More sophisticated emotion heuristics
        # Confident: Large face, centered, stable position, looking straight
        # Fearful: Small face, off-center, rapid movement
        # Sad: Looking down, slower movement
        # Angry: Strong movement, leaning forward (larger face)
        # Happy: Centered, moderate movement
        # Neutral: Default case
        
        # Weighted emotion calculations
        confidence_score = (face_ratio * 3) + ((1 - normalized_distance) * 2) + (1 - min(avg_motion / 200, 1))
        fearful_score = ((1 - face_ratio) * 2) + normalized_distance + min(avg_motion / 100, 1)
        sad_score = vertical_position * 3 + (1 - face_ratio) + (1 - min(avg_motion / 150, 1))
        angry_score = min(avg_motion / 80, 1) * 2 + face_ratio + normalized_distance
        happy_score = ((1 - normalized_distance) * 1.5) + (min(avg_motion / 120, 1) * 1.5) + (face_ratio * 0.5)
        
        # Create weighted probabilities for smoother transitions
        emotion_scores = {
            'confident': confidence_score,
            'fearful': fearful_score,
            'sad': sad_score,
            'angry': angry_score,
            'happy': happy_score,
            'neutral': 1.0  # Baseline for neutral
        }
        
        # Determine the emotion with highest score
        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        predicted_emotion = max_emotion[0]
        
        # Add stability to emotion predictions to prevent rapid changes
        if self.last_emotion == predicted_emotion:
            self.emotion_stability += 1
        else:
            self.emotion_stability -= 1
            
        # Only change emotion after stability threshold or no previous emotion
        if self.last_emotion is None or self.emotion_stability <= 0 or self.emotion_stability >= 3:
            self.last_emotion = predicted_emotion
            self.emotion_stability = max(0, min(5, self.emotion_stability))  # Keep within bounds
        
        # Return the stable emotion prediction
        return self.last_emotion, frame

    def get_emotion_color(self, emotion):
        """Get color based on detected emotion"""
        emotion_colors = {
            'neutral': 'blue',
            'happy': 'green',
            'sad': 'purple',
            'angry': 'red',
            'fearful': 'orange',
            'confident': 'teal'
        }
        return emotion_colors.get(emotion, 'gray')
