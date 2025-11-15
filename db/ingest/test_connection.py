from get_connection import get_db_connection

try:
    conn = get_db_connection()
    print("✅ Successfully connected to PostgreSQL!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)
    
    