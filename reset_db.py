#!/usr/bin/env python3
"""
Reset database - Drop all tables and recreate them
"""
from database import Base, engine
from models import User, Field, FieldThumbnail, VISnapshot, VITimeSeries, ImportExportLog

def reset_database():
    """Drop all tables and recreate them"""
    try:
        print("🗑️  Dropping all existing tables...")
        Base.metadata.drop_all(bind=engine)
        print("✅ All tables dropped successfully!")
        
        print("🔨 Creating new tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")
        
        # Print table information
        print("\n📋 Created tables:")
        for table in Base.metadata.tables.keys():
            print(f"   - {table}")
            
        print("\n🎉 Database reset completed!")
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        raise

if __name__ == "__main__":
    reset_database()
