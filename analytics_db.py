import sqlite3
from datetime import datetime

DB_PATH = "analytics.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  username TEXT,
                  action TEXT,
                  value REAL)''')
    conn.commit()
    conn.close()

def log_event(username, action, value=0.0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO events (username, action, value) VALUES (?, ?, ?)",
              (username, action, value))
    conn.commit()
    conn.close()

def get_total_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT username) FROM events")
    result = c.fetchone()[0]
    conn.close()
    return result

def get_questions_today():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM events WHERE action='query' AND date(timestamp) = date('now')")
    result = c.fetchone()[0]
    conn.close()
    return result

def get_avg_score():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT AVG(value) FROM events WHERE action='evaluation'")
    result = c.fetchone()[0]
    conn.close()
    return result if result else 0.0

def get_daily_requests():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Group by last 7 days
    c.execute('''SELECT date(timestamp), COUNT(*) 
                 FROM events 
                 WHERE action='query' 
                 GROUP BY date(timestamp) 
                 ORDER BY date(timestamp) DESC LIMIT 7''')
    result = c.fetchall()
    conn.close()
    return result

def get_user_activity():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT username, COUNT(*) 
                 FROM events 
                 WHERE action='query' 
                 GROUP BY username''')
    result = c.fetchall()
    conn.close()
    return result
