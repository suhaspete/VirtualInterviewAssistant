import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
import time
import sqlite3
import os
import requests
from bs4 import BeautifulSoup
import re
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import custom modules
from resource_recommender import ResourceRecommender
from emotion_detection import EmotionDetector
from voice_interaction import VoiceInteractionSystem

class AdaptiveInterviewSystem:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        if not GEMINI_API_KEY:
            st.error("API Key is missing! Please check your .env file.")
            st.stop()

        # Gemini API Configuration
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize SQLite Database
        self.conn = sqlite3.connect('interview_database.db', check_same_thread=False)
        self.create_tables()
        
        # Initialize helper systems
        self.resource_recommender = ResourceRecommender(self.model)
        self.emotion_detector = EmotionDetector()
        self.voice_system = VoiceInteractionSystem()
        
        # Confidence levels mapping
        self.confidence_levels = {
            'beginner': {'level': 1, 'complexity': 'basic and straightforward', 'depth': 'entry-level'},
            'intermediate': {'level': 2, 'complexity': 'moderately challenging', 'depth': 'mid-level'},
            'advanced': {'level': 3, 'complexity': 'advanced and complex', 'depth': 'expert-level'}
        }
        
        # Emotion tracking
        self.emotion_history = []

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title TEXT,
                confidence_level TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER,
                round_number INTEGER,
                question TEXT,
                answer TEXT,
                ai_evaluation TEXT,
                ml_score REAL,
                emotions TEXT,
                FOREIGN KEY (interview_id) REFERENCES interviews(id)
            )
        ''')
        
        # Add new table for storing resource recommendations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER,
                topic TEXT,
                resource_title TEXT,
                resource_url TEXT,
                resource_description TEXT,
                FOREIGN KEY (interview_id) REFERENCES interviews(id)
            )
        ''')
        
        self.conn.commit()

    def calculate_ml_score(self, question, answer):
        if not answer:
            return 0.0
        corpus = [question, answer]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity_matrix[0][0])

    def generate_adaptive_questions(self, job_title, confidence_level):
        config = self.confidence_levels[confidence_level]
        prompt = f"""
        Generate a professional {config['complexity']} interview 
        question for a {job_title} role suitable for {config['depth']} candidates.
        """
        response = self.model.generate_content(prompt)
        return response.text

    def evaluate_answer(self, job_title, question, answer, confidence_level):
        config = self.confidence_levels[confidence_level]
        evaluation_prompt = f"""
        Evaluate this {config['depth']} answer for a {job_title} role:
        Question: {question}
        Answer: {answer}
        
        Provide constructive feedback that helps the candidate improve.
        """
        response = self.model.generate_content(evaluation_prompt)
        return response.text
    
    def generate_resource_recommendations(self, job_title, evaluations):
        """
        Generate personalized resource recommendations based on interview evaluations
        
        Args:
            job_title: The job title for the interview
            evaluations: List of evaluation texts from the interview
            
        Returns:
            List of dictionaries with topic and resource information
        """
        # Identify weak topics
        weak_topics = self.resource_recommender.identify_weak_topics(job_title, evaluations)
        
        # Find resources for each weak topic
        topic_resources = []
        for topic_data in weak_topics:
            topic = topic_data['topic']
            resources = self.resource_recommender.find_learning_resources(topic, job_title)
            topic_resources.append({
                'topic': topic,
                'reason': topic_data['reason'],
                'resources': resources
            })
        
        return topic_resources
    
    def save_resource_recommendations(self, interview_id, topic_resources):
        """
        Save resource recommendations to the database
        
        Args:
            interview_id: ID of the interview
            topic_resources: List of dictionaries containing topic and resources data
        """
        cursor = self.conn.cursor()
        
        for topic_data in topic_resources:
            topic = topic_data['topic']
            for resource in topic_data['resources']:
                cursor.execute('''
                    INSERT INTO resource_recommendations 
                    (interview_id, topic, resource_title, resource_url, resource_description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    interview_id, 
                    topic, 
                    resource['title'], 
                    resource['url'], 
                    resource['description']
                ))
        
        self.conn.commit()

    def save_interview_data(self, job_title, confidence_level, rounds):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO interviews (job_title, confidence_level) VALUES (?, ?)', (job_title, confidence_level))
        interview_id = cursor.lastrowid
        
        for round_data in rounds:
            ml_score = self.calculate_ml_score(round_data['question'], round_data['answer'])
            emotions_json = json.dumps(round_data.get('emotions', []))
            
            cursor.execute('''
                INSERT INTO interview_rounds 
                (interview_id, round_number, question, answer, ai_evaluation, ml_score, emotions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                interview_id, 
                rounds.index(round_data) + 1, 
                round_data['question'], 
                round_data['answer'], 
                round_data['evaluation'], 
                ml_score,
                emotions_json
            ))
        
        self.conn.commit()
        return interview_id

def initialize_session_state():
    base_keys = {
        'stage': 'setup', 
        'round': 1, 
        'answers': [], 
        'job_title': '', 
        'confidence_level': 'beginner',
        'resource_recommendations': [],
        'current_question': '',
        'voice_enabled': False,
        'camera_enabled': False,
        'emotions': [],
        'current_emotion': None
    }
    
    for key, default_value in base_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    st.set_page_config(page_title="AI Interview Assistant", layout="wide", page_icon="🎓")
    
    initialize_session_state()
    interview_system = AdaptiveInterviewSystem()
    
    # Custom CSS for emotion indicators
    st.markdown("""
    <style>
    .emotion-indicator {
        padding: 5px 15px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 5px;
        display: inline-block;
    }
    .neutral { background-color: #2196F3; }
    .happy { background-color: #4CAF50; }
    .sad { background-color: #9C27B0; }
    .angry { background-color: #F44336; }
    .fearful { background-color: #FF9800; }
    .confident { background-color: #009688; }
    
    .resource-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    
    .topic-header {
        font-size: 18px;
        font-weight: bold;
        color: #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 Adaptive AI Interview Assistant")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Interview Settings")
        
        # Voice interaction toggle
        voice_toggle = st.checkbox("Enable Voice Interaction", value=st.session_state.voice_enabled)
        if voice_toggle != st.session_state.voice_enabled:
            st.session_state.voice_enabled = voice_toggle
            st.rerun()
            
        # Camera toggle for emotion detection
        camera_toggle = st.checkbox("Enable Emotion Detection", value=st.session_state.camera_enabled)
        if camera_toggle != st.session_state.camera_enabled:
            st.session_state.camera_enabled = camera_toggle
            st.rerun()
            
        st.divider()
        
        # Display current emotion if camera is enabled
        if st.session_state.camera_enabled and st.session_state.current_emotion:
            st.subheader("Current Emotion")
            emotion_color = interview_system.emotion_detector.get_emotion_color(st.session_state.current_emotion)
            st.markdown(f"""
            <div class="emotion-indicator {st.session_state.current_emotion}">
                {st.session_state.current_emotion.upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Display emotion history
            if len(st.session_state.emotions) > 0:
                st.subheader("Emotion History")
                emotion_counts = {}
                for emotion in st.session_state.emotions:
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1
                    else:
                        emotion_counts[emotion] = 1
                
                for emotion, count in emotion_counts.items():
                    emotion_color = interview_system.emotion_detector.get_emotion_color(emotion)
                    st.markdown(f"""
                    <div class="emotion-indicator {emotion}">
                        {emotion.upper()}: {count}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Camera feed for emotion detection
    if st.session_state.camera_enabled:
        # Create a placeholder for the camera feed
        camera_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            st.session_state.camera_enabled = False
        else:
            # Read the first frame
            ret, frame = cap.read()
            
            if ret:
                # Process the frame for emotion detection
                emotion, processed_frame = interview_system.emotion_detector.analyze_emotion(frame)
                
                if emotion:
                    st.session_state.current_emotion = emotion
                    if len(st.session_state.emotions) < 100:  # Limit history
                        st.session_state.emotions.append(emotion)
                
                # Convert from BGR to RGB
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the processed frame
                camera_placeholder.image(processed_frame_rgb, channels="RGB", caption="Camera Feed (Emotion Detection)")
            
            # Release the capture when done
            cap.release()
    
    # Interview stage handling
    if st.session_state.stage == 'setup':
        st.subheader("Let's prepare for your interview")
        
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Enter Job Title:", placeholder="Software Engineer, Data Scientist...")
        with col2:
            confidence_level = st.select_slider("Confidence Level", options=['beginner', 'intermediate', 'advanced'])
        
        if st.button("Begin Interview", use_container_width=True):
            if not job_title:
                st.error("Please enter a job title.")
                return
            
            st.session_state.job_title = job_title
            st.session_state.confidence_level = confidence_level
            st.session_state.stage = 'interview'
            
            # Generate first question
            question = interview_system.generate_adaptive_questions(job_title, confidence_level)
            st.session_state.current_question = question
            
            # If voice enabled, speak the question
            if st.session_state.voice_enabled:
                interview_system.voice_system.speak_text(f"Welcome to your {job_title} interview. Here's your first question: {question}")
                
            st.rerun()
    
    elif st.session_state.stage == 'interview':
        st.subheader(f"Round {st.session_state.round}")
        
        # Display the current question
        question = st.session_state.current_question
        st.markdown(f"**Question {st.session_state.round}:**\n\n{question}")
        
        # Answer input method (text or voice)
        if st.session_state.voice_enabled:
            # Voice input option
            if st.button("Answer with Voice"):
                answer_text = interview_system.voice_system.listen_for_speech(timeout=20)
                if answer_text:
                    st.success(f"Detected answer: {answer_text}")
                    st.session_state.answer_text = answer_text
        
        # Text input is always available
        answer = st.text_area(
            "Type your answer:" if not st.session_state.voice_enabled else "Review and edit your answer if needed:",
            value=st.session_state.get('answer_text', ''),
            height=150
        )
        
        if st.button(f"Submit Answer for Round {st.session_state.round}", use_container_width=True):
            if not answer:
                st.error("Please provide an answer before submitting.")
                return
                
            with st.spinner("Evaluating your answer..."):
                # Evaluate the answer
                evaluation = interview_system.evaluate_answer(
                    st.session_state.job_title, 
                    question, 
                    answer, 
                    st.session_state.confidence_level
                )
                
                # Store answer data with emotion information
                st.session_state.answers.append({
                    'question': question, 
                    'answer': answer, 
                    'evaluation': evaluation,
                    'emotions': st.session_state.emotions[-20:] if st.session_state.emotions else []
                })
                
                # If voice enabled, speak the evaluation summary
                if st.session_state.voice_enabled:
                    # Simplify evaluation for speech (first two sentences)
                    simple_eval = ". ".join(evaluation.split(".")[:2]) + "."
                    interview_system.voice_system.speak_text(simple_eval)
                
                # Reset emotion tracking for next question
                st.session_state.emotions = []
                
                # Check if we should move to next round or complete the interview
                if st.session_state.round < 3:
                    st.session_state.round += 1
                    
                    # Generate next question
                    next_question = interview_system.generate_adaptive_questions(
                        st.session_state.job_title, 
                        st.session_state.confidence_level
                    )
                    st.session_state.current_question = next_question
                    
                    # Clear the answer text
                    if 'answer_text' in st.session_state:
                        del st.session_state.answer_text
                else:
                    st.session_state.stage = 'complete'
                    
                    # Generate resource recommendations
                    with st.spinner("Generating personalized learning resources..."):
                        evaluations = [round_data['evaluation'] for round_data in st.session_state.answers]
                        topic_resources = interview_system.generate_resource_recommendations(
                            st.session_state.job_title, 
                            evaluations
                        )
                        st.session_state.resource_recommendations = topic_resources
                
                st.rerun()
    
    elif st.session_state.stage == 'complete':
        st.success("🎉 Interview Completed Successfully!")
        
        # Save interview data to database
        interview_id = interview_system.save_interview_data(
            st.session_state.job_title, 
            st.session_state.confidence_level, 
            st.session_state.answers
        )
        
        # Display all round details
        for i, round_data in enumerate(st.session_state.answers, 1):
            with st.expander(f"Round {i} Details"):
                st.markdown(f"**Question:**\n\n{round_data['question']}")
                st.markdown(f"**Your Answer:**\n\n{round_data['answer']}")
                st.markdown(f"**AI Evaluation:**\n\n{round_data['evaluation']}")
                
                ml_score = interview_system.calculate_ml_score(round_data['question'], round_data['answer'])
                st.metric(f"Round {i} Similarity Score", f"{ml_score:.2f}")
                
                # Show emotions during this round if available
                if round_data.get('emotions'):
                    st.subheader("Your Emotions During This Round")
                    emotion_counts = {}
                    for emotion in round_data['emotions']:
                        if emotion in emotion_counts:
                            emotion_counts[emotion] += 1
                        else:
                            emotion_counts[emotion] = 1
                    
                    emotion_html = ""
                    for emotion, count in emotion_counts.items():
                        emotion_html += f'<div class="emotion-indicator {emotion}">{emotion.upper()}: {count}</div>'
                    
                    st.markdown(f"""<div>{emotion_html}</div>""", unsafe_allow_html=True)
        
        # Display personalized resource recommendations
        st.subheader("📚 Personalized Learning Resources")
        st.write("Based on your interview performance, we've identified areas for improvement and found resources to help you grow:")
        
        # Save recommendations to database
        interview_system.save_resource_recommendations(interview_id, st.session_state.resource_recommendations)
        
        # Display each topic and its resources
        for topic_data in st.session_state.resource_recommendations:
            with st.expander(f"🔍 Topic: {topic_data['topic']}"):
                st.markdown(f"**Why this matters:** {topic_data['reason']}")
                st.markdown("**Recommended Resources:**")
                
                for i, resource in enumerate(topic_data['resources'], 1):
                    # Ensure URL has proper format
                    url = resource['url']
                    if not url.startswith('http://') and not url.startswith('https://'):
                        url = 'https://' + url.lstrip('/')
                    
                    # Create a card for each resource
                    st.markdown(f"""
                    <div class="resource-card">
                        <p class="topic-header">{i}. {resource['title']}</p>
                        <a href="{url}" target="_blank" rel="noopener noreferrer">
                            <button style="background-color: #4CAF50; color: white; padding: 8px 16px; 
                            border: none; border-radius: 4px; cursor: pointer; margin: 10px 0;">
                                Visit Resource
                            </button>
                        </a>
                        <p>{resource['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Option to start a new interview
        if st.button("Start New Interview", use_container_width=True):
            # Clear session state and restart
            for key in list(st.session_state.keys()):
                if key not in ['voice_enabled', 'camera_enabled']:
                    del st.session_state[key]
            
            initialize_session_state()
            st.rerun()

if __name__ == "__main__":
    main()
