#!/usr/bin/env python3
"""
Check database tables and columns
"""
import psycopg2

def check_database():
    """Check database tables and columns"""
    try:
        conn = psycopg2.connect('postgresql://postgres:1234@localhost:5432/DATA')
        cur = conn.cursor()
        
        print("📋 Checking database tables...")
        
        # Get all tables
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = cur.fetchall()
        
        if not tables:
            print("❌ No tables found!")
            return
            
        print(f"✅ Found {len(tables)} tables:")
        for table in tables:
            print(f"   - {table[0]}")
            
        # Check users table specifically
        print("\n🔍 Checking users table columns:")
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users';")
        columns = cur.fetchall()
        
        if columns:
            for col in columns:
                print(f"   - {col[0]} ({col[1]})")
        else:
            print("❌ Users table not found or has no columns!")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    check_database()
