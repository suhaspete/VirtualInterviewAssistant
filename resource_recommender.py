import requests
from bs4 import BeautifulSoup
import re
import json
import streamlit as st
import google.generativeai as genai

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
