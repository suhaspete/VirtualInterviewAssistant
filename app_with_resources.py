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

class ResourceRecommender:
    def __init__(self, model):
        """
        Initialize the ResourceRecommender with a Gemini model
        
        Args:
            model: A configured Gemini model instance
        """
        self.model = model
    
    def identify_weak_topics(self, job_title, evaluations):
        """
        Analyze AI evaluations to identify weak topics that need improvement
        
        Args:
            job_title: The job title for the interview
            evaluations: List of evaluation texts from the interview
            
        Returns:
            List of dictionaries with 'topic' and 'reason' keys
        """
        all_evaluations = "\n".join([eval_text for eval_text in evaluations])
        
        prompt = f"""
        Based on the following interview evaluations for a {job_title} role, identify the top 3 specific 
        technical topics or skills where the candidate needs improvement. For each topic, provide:
        1. The exact topic name (be specific, e.g., 'React Hooks' instead of just 'React')
        2. A brief explanation of why this is a weak area
        
        Format your response as a JSON array with objects containing 'topic' and 'reason' keys.
        
        Evaluations:
        {all_evaluations}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Extract JSON from response
            response_text = response.text
            # Find JSON array in the response
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            if json_match:
                topics = json.loads(json_match.group())
                return topics
            else:
                # Fallback if JSON parsing fails
                return [{"topic": "General interview skills", "reason": "Based on overall evaluation"}]
        except Exception as e:
            st.error(f"Error identifying weak topics: {str(e)}")
            return [{"topic": "General interview skills", "reason": "Based on overall evaluation"}]
    
    def find_learning_resources(self, topic, job_title):
        """
        Find learning resources for a specific topic using web scraping and Gemini API
        
        Args:
            topic: The specific topic to find resources for
            job_title: The job title for context
            
        Returns:
            List of dictionaries with 'title', 'url', and 'description' keys
        """
        # First, use Gemini to get better search queries
        search_prompt = f"""
        Generate 2 specific search queries to find learning resources for someone 
        who needs to improve their knowledge of '{topic}' for a {job_title} role.
        Make the queries specific and targeted to find high-quality educational content.
        Format your response as a JSON array of strings.
        """
        
        try:
            response = self.model.generate_content(search_prompt)
            response_text = response.text
            # Extract JSON array from response
            json_match = re.search(r'\[\s*".*?"\s*\]', response_text, re.DOTALL)
            if json_match:
                search_queries = json.loads(json_match.group())
            else:
                search_queries = [f"best resources to learn {topic} for {job_title}"]
        except Exception:
            search_queries = [f"best resources to learn {topic} for {job_title}"]
        
        # Collect resources from web search
        all_resources = []
        
        for query in search_queries[:1]:  # Use just the first query to avoid too many requests
            try:
                # Format query for URL
                formatted_query = query.replace(' ', '+')
                url = f"https://www.google.com/search?q={formatted_query}"
                
                # Send request with headers to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract search results
                    search_results = soup.find_all('div', class_='g')
                    
                    for result in search_results[:5]:  # Get top 5 results
                        title_element = result.find('h3')
                        link_element = result.find('a')
                        
                        if title_element and link_element:
                            title = title_element.text
                            link = link_element.get('href')
                            
                            # Extract actual URL from Google's redirect URL
                            if link and link.startswith('/url?q='):
                                link = link.split('/url?q=')[1].split('&')[0]
                            
                            # Skip if link is None or doesn't start with http
                            if not link or not link.startswith('http'):
                                continue
                                
                            # Get a short description
                            description_element = result.find('div', class_='VwiC3b')
                            description = description_element.text if description_element else "No description available"
                            
                            all_resources.append({
                                'title': title,
                                'url': link,
                                'description': description[:200] + '...' if len(description) > 200 else description
                            })
            except Exception as e:
                st.warning(f"Error during web scraping: {str(e)}")
                continue
        
        # If web scraping failed or returned no results, use Gemini to suggest resources
        if not all_resources:
            fallback_prompt = f"""
            Provide 3 specific learning resources for someone who needs to improve their knowledge of '{topic}' 
            for a {job_title} role. For each resource, include:
            1. Title of the resource
            2. URL (if you don't know the exact URL, suggest a general domain like "coursera.org" or "udemy.com")
            3. A brief description of what the resource covers
            
            Format your response as a JSON array with objects containing 'title', 'url', and 'description' keys.
            """
            
            try:
                response = self.model.generate_content(fallback_prompt)
                response_text = response.text
                # Find JSON array in the response
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
                if json_match:
                    all_resources = json.loads(json_match.group())
                else:
                    # Create a generic resource if all else fails
                    all_resources = [{
                        'title': f"Learning {topic} for {job_title}",
                        'url': f"https://www.google.com/search?q=learn+{topic.replace(' ', '+')}",
                        'description': f"Search for resources about {topic} relevant to {job_title} roles."
                    }]
            except Exception as e:
                st.error(f"Error generating resource recommendations: {str(e)}")
                all_resources = [{
                    'title': f"Learning {topic} for {job_title}",
                    'url': f"https://www.google.com/search?q=learn+{topic.replace(' ', '+')}",
                    'description': f"Search for resources about {topic} relevant to {job_title} roles."
                }]
        
        return all_resources[:3]  # Return top 3 resources

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
        
        # Initialize resource recommender
        self.resource_recommender = ResourceRecommender(self.model)
        
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
            cursor.execute('''
                INSERT INTO interview_rounds 
                (interview_id, round_number, question, answer, ai_evaluation, ml_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (interview_id, rounds.index(round_data) + 1, round_data['question'], round_data['answer'], round_data['evaluation'], ml_score))
        
        self.conn.commit()
        return interview_id

def initialize_session_state():
    base_keys = {'stage': 'setup', 'round': 1, 'answers': [], 'job_title': '', 'confidence_level': 'beginner', 'resource_recommendations': []}
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
        
        # Generate resource recommendations
        with st.spinner("Analyzing your interview and finding personalized learning resources..."):
            # Extract all evaluations
            evaluations = [round_data['evaluation'] for round_data in st.session_state.answers]
            
            # Generate resource recommendations
            topic_resources = interview_system.generate_resource_recommendations(st.session_state.job_title, evaluations)
            
            # Save recommendations to database
            interview_system.save_resource_recommendations(interview_id, topic_resources)
            
            # Store in session state
            st.session_state.resource_recommendations = topic_resources
        
        # Display resource recommendations
        st.subheader("📚 Personalized Learning Resources")
        st.write("Based on your interview, we've identified areas for improvement and found resources to help you grow:")
        
        for topic_data in st.session_state.resource_recommendations:
            with st.expander(f"🔍 Topic: {topic_data['topic']}"):
                st.write(f"**Why this matters:** {topic_data['reason']}")
                st.write("**Recommended Resources:**")
                
                for i, resource in enumerate(topic_data['resources'], 1):
                    # Using HTML with target="_blank" to open links in a new tab
                    st.markdown(f"<h4>{i}. <a href='{resource['url']}' target='_blank'>{resource['title']}</a></h4>", unsafe_allow_html=True)
                    st.write(resource['description'])
                    st.write("---")
        
        if st.button("Start New Interview"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
