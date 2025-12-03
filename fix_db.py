#!/usr/bin/env python3
"""
Fix database by adding missing columns
"""
import psycopg2

def fix_database():
    """Add missing columns to users table"""
    try:
        conn = psycopg2.connect('postgresql://postgres:1234@localhost:5432/DATA')
        cur = conn.cursor()
        
        print("🔧 Adding missing columns to users table...")
        
        # Add is_active column
        try:
            cur.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE;")
            print("✅ Added is_active column")
        except psycopg2.errors.DuplicateColumn:
            print("ℹ️ is_active column already exists")
        
        conn.commit()
        
        # Check final columns
        print("\n📋 Final users table columns:")
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'users';")
        columns = cur.fetchall()
        
        for col in columns:
            print(f"   - {col[0]} ({col[1]})")
            
        conn.close()
        print("\n✅ Database fix completed!")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        raise

if __name__ == "__main__":
    fix_database()
