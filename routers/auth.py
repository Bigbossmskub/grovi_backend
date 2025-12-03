from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import User
from schemas import UserCreate, UserLogin, UserResponse, Token
from auth import authenticate_user, create_access_token, get_password_hash, get_current_user
from config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])

def calculate_age(date_of_birth: datetime) -> int:
    """Calculate age from date of birth"""
    today = datetime.today()
    age = today.year - date_of_birth.year
    if today.month < date_of_birth.month or (today.month == date_of_birth.month and today.day < date_of_birth.day):
        age -= 1
    return age

@router.post("/register", response_model=Token)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Calculate age if date of birth is provided
    age = None
    if user_data.date_of_birth:
        age = calculate_age(user_data.date_of_birth)
    
    # Create new user
    password_hash = get_password_hash(user_data.password)
    db_user = User(
        name=user_data.name,
        username=user_data.username,
        email=user_data.email,
        password_hash=password_hash,
        date_of_birth=user_data.date_of_birth,
        age=age
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(db_user.id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(db_user)
    }

@router.post("/login", response_model=Token)
def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    user = authenticate_user(db, login_data.username_or_email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user)
    }

@router.get("/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse.model_validate(current_user)
