import sqlite3

def create_tables(conn):
    """
    Create all necessary database tables if they don't exist
    
    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()
    
    # Interviews table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT,
            confidence_level TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Interview rounds table
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
    
    # Resource recommendations table
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
    
    conn.commit()

def save_resource_recommendations(conn, interview_id, topic_resources):
    """
    Save resource recommendations to the database
    
    Args:
        conn: SQLite database connection
        interview_id: ID of the interview
        topic_resources: List of dictionaries containing topic and resources data
    """
    cursor = conn.cursor()
    
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
    
    conn.commit()
