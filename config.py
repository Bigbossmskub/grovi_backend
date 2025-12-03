from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:1234@localhost:5432/DATA"
    secret_key: str = "grovi-secret-key-2024-crop-monitoring-system"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    gee_service_account_email: str = "gee-backend@woven-invention-465809-d9.iam.gserviceaccount.com"
    gee_project_id: str = "woven-invention-465809-d9"
    gee_key_file: str = "woven-invention-465809-d9-1bd4e2ff20f3.json"

    class Config:
        env_file = ".env"

settings = Settings()
