import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
import time
import sqlite3
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        
        # Confidence levels mapping
        self.confidence_levels = {
            'beginner': {'level': 1, 'complexity': 'basic and straightforward', 'depth': 'entry-level'},
            'intermediate': {'level': 2, 'complexity': 'moderately challenging', 'depth': 'mid-level'},
            'advanced': {'level': 3, 'complexity': 'advanced and complex', 'depth': 'expert-level'}
        }

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
        """
        response = self.model.generate_content(evaluation_prompt)
        return response.text

    def save_interview_data(self, job_title, confidence_level, rounds):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO interviews (job_title, confidence_level) VALUES (?, ?)', (job_title, confidence_level))
        interview_id = cursor.lastrowid
        
        for round_data in rounds:
            ml_score = self.calculate_ml_score(round_data['question'], round_data['answer'])
            cursor.execute('''
                INSERT INTO interview_rounds 
                (interview_id, round_number, question, answer, ai_evaluation, ml_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (interview_id, rounds.index(round_data) + 1, round_data['question'], round_data['answer'], round_data['evaluation'], ml_score))
        
        self.conn.commit()
        return interview_id

def initialize_session_state():
    base_keys = {'stage': 'setup', 'round': 1, 'answers': [], 'job_title': '', 'confidence_level': 'beginner'}
    for key, default_value in base_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    initialize_session_state()
    interview_system = AdaptiveInterviewSystem()
    
    st.title("🚀 Adaptive AI Interview Assistant")
    
    if st.session_state.stage == 'setup':
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
            st.rerun()
    
    if st.session_state.stage == 'interview':
        st.subheader(f"Round {st.session_state.round}")
        question = interview_system.generate_adaptive_questions(st.session_state.job_title, st.session_state.confidence_level)
        st.write("**Question:**", question)
        answer = st.text_area(f"Answer for Question {st.session_state.round}:", height=200)
        
        if st.button(f"Submit Answer for Round {st.session_state.round}"):
            evaluation = interview_system.evaluate_answer(st.session_state.job_title, question, answer, st.session_state.confidence_level)
            st.session_state.answers.append({'question': question, 'answer': answer, 'evaluation': evaluation})
            
            if st.session_state.round < 3:
                st.session_state.round += 1
            else:
                st.session_state.stage = 'complete'
            
            st.rerun()
    
    if st.session_state.stage == 'complete':
        st.success("🎉 Interview Completed Successfully!")
        interview_id = interview_system.save_interview_data(st.session_state.job_title, st.session_state.confidence_level, st.session_state.answers)
        
        for i, round_data in enumerate(st.session_state.answers, 1):
            with st.expander(f"Round {i} Details"):
                st.write("**Question:**", round_data['question'])
                st.write("**Your Answer:**", round_data['answer'])
                st.write("**AI Evaluation:**", round_data['evaluation'])
                ml_score = interview_system.calculate_ml_score(round_data['question'], round_data['answer'])
                st.metric(f"Round {i} Similarity Score", f"{ml_score:.2f}")
        
        st.info(f"Interview saved with ID: {interview_id}")
        
        if st.button("Start New Interview"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
