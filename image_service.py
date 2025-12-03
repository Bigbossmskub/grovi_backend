import os
import uuid
import base64
import requests
from pathlib import Path
from typing import Optional
from PIL import Image
from io import BytesIO

class ImageService:
    def __init__(self):
        """Initialize Image Service"""
        self.storage_dir = Path("backend/storage/vi_images")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def save_base64_image(self, base64_data: str, filename_prefix: str = "vi") -> str:
        """Save base64 image to file and return file path"""
        try:
            # Remove data:image/png;base64, prefix if present
            if "data:image" in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            
            # Generate unique filename
            filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
            file_path = self.storage_dir / filename
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            
            # Return relative path from backend root
            return f"storage/vi_images/{filename}"
            
        except Exception as e:
            print(f"Error saving base64 image: {e}")
            return None
    
    def save_url_image(self, image_url: str, filename_prefix: str = "vi") -> Optional[str]:
        """Download image from URL and save to file"""
        try:
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Generate unique filename
            filename = f"{filename_prefix}_{uuid.uuid4().hex}.png"
            file_path = self.storage_dir / filename
            
            # Process and save image
            image = Image.open(BytesIO(response.content))
            image.save(file_path, 'PNG')
            
            # Return relative path from backend root
            return f"storage/vi_images/{filename}"
            
        except Exception as e:
            print(f"Error saving URL image: {e}")
            return None
    
    def get_image_url(self, file_path: str) -> str:
        """Convert file path to accessible URL"""
        # For development, return local file path
        # In production, this would return a proper URL to serve the image
        return f"/images/{file_path}"
    
    def delete_image(self, file_path: str) -> bool:
        """Delete image file"""
        try:
            full_path = Path("backend") / file_path
            if full_path.exists():
                full_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
    
    def cleanup_old_images(self, days_old: int = 30):
        """Clean up images older than specified days"""
        try:
            import time
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            
            for file_path in self.storage_dir.glob("*.png"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    print(f"Deleted old image: {file_path}")
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Global instance
image_service = ImageService()
