import sqlite3

def update_database_schema():
    """Add emotions column to interview_rounds table if it doesn't exist"""
    conn = sqlite3.connect('interview_database.db')
    cursor = conn.cursor()
    
    # Check if emotions column exists in interview_rounds table
    cursor.execute("PRAGMA table_info(interview_rounds)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Add emotions column if it doesn't exist
    if 'emotions' not in columns:
        print("Adding 'emotions' column to interview_rounds table...")
        cursor.execute("ALTER TABLE interview_rounds ADD COLUMN emotions TEXT")
        conn.commit()
        print("Database schema updated successfully!")
    else:
        print("Emotions column already exists in interview_rounds table.")
    
    conn.close()

if __name__ == "__main__":
    update_database_schema()
