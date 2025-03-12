import streamlit as st
import speech_recognition as sr
import pyttsx3
import threading
import queue
import time

class VoiceInteractionSystem:
    """A class for handling voice interactions (text-to-speech and speech-to-text)"""
    
    def __init__(self):
        """Initialize the voice interaction system"""
        # Initialize text to speech engine
        self.engine = pyttsx3.init()
        
        # Set voice properties
        voices = self.engine.getProperty('voices')
        # Use a female voice if available
        for voice in voices:
            if 'female' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        self.engine.setProperty('rate', 175)  # Speed of speech
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300  # Increased sensitivity
        
        # Queue for async processing
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        
        # Start the speaking thread
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()
    
    def _process_speech_queue(self):
        """Background thread to process the speech queue"""
        while True:
            if not self.speech_queue.empty():
                text = self.speech_queue.get()
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
                self.speech_queue.task_done()
            else:
                time.sleep(0.1)
    
    def speak_text(self, text, wait=False):
        """Convert text to speech"""
        self.speech_queue.put(text)
        
        if wait:
            # Wait until speaking is done
            self.speech_queue.join()
            while self.is_speaking:
                time.sleep(0.1)
    
    def listen_for_speech(self, timeout=5):
        """
        Listen for speech and convert to text
        
        Args:
            timeout: Maximum seconds to listen for
            
        Returns:
            Text of recognized speech or None if not recognized
        """
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                # Listen for speech
                audio = self.recognizer.listen(source, timeout=timeout)
                
                # Convert speech to text
                text = self.recognizer.recognize_google(audio)
                return text
                
            except sr.WaitTimeoutError:
                st.warning("No speech detected within the timeout period.")
                return None
            except sr.UnknownValueError:
                st.warning("Could not understand audio.")
                return None
            except sr.RequestError:
                st.error("Could not request results from speech recognition service.")
                return None
            except Exception as e:
                st.error(f"Error in speech recognition: {str(e)}")
                return None
