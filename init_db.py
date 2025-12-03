#!/usr/bin/env python3
"""
Initialize database with tables
"""
from database import Base, engine
from models import User, Field, FieldThumbnail, VISnapshot, VITimeSeries, ImportExportLog

def init_database():
    """Initialize database with all tables"""
    try:
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully!")
        
        # Print table information
        print("\nCreated tables:")
        for table in Base.metadata.tables.keys():
            print(f"  - {table}")
            
    except Exception as e:
        print(f"Error creating database tables: {e}")
        raise

if __name__ == "__main__":
    init_database()
