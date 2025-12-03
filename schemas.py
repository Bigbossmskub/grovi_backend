from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

# User schemas
class UserBase(BaseModel):
    name: str
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str
    date_of_birth: Optional[datetime] = None

class UserResponse(UserBase):
    id: UUID
    age: Optional[int] = None
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Field schemas
class FieldBase(BaseModel):
    name: str = "แปลง A1"
    crop_type: Optional[str] = "ข้าวหอมมะลิ"
    variety: Optional[str] = "ข้าวหอมมะลิ"
    planting_season: Optional[str] = None
    planting_date: Optional[datetime] = None

class FieldCreate(FieldBase):
    geometry: Dict[str, Any]  # GeoJSON polygon
    
class FieldUpdate(FieldBase):
    geometry: Optional[Dict[str, Any]] = None

class FieldResponse(FieldBase):
    id: UUID
    user_id: UUID
    geometry: Dict[str, Any]
    area_m2: float
    centroid_lat: float
    centroid_lng: float
    address: Optional[str] = None
    variety: Optional[str] = None
    planting_season: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Thumbnail schemas
class ThumbnailCreate(BaseModel):
    field_id: UUID
    image_data: str

class ThumbnailResponse(BaseModel):
    id: UUID
    field_id: UUID
    image_data: str
    created_at: datetime

    class Config:
        from_attributes = True

# VI Snapshot schemas
class VISnapshotCreate(BaseModel):
    field_id: UUID
    vi_type: str
    snapshot_date: datetime
    mean_value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    overlay_url: Optional[str] = None
    grid_data: Optional[Dict[str, Any]] = None
    analysis_message: Optional[str] = None

class VISnapshotResponse(BaseModel):
    id: UUID
    field_id: UUID
    user_id: UUID
    vi_type: str
    snapshot_date: datetime
    mean_value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    overlay_data: Optional[str] = None
    status_message: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# VI TimeSeries schemas
class VITimeSeriesCreate(BaseModel):
    field_id: UUID
    vi_type: str
    measurement_date: datetime
    vi_value: float

class VITimeSeriesResponse(BaseModel):
    id: UUID
    field_id: UUID
    vi_type: str
    measurement_date: datetime
    vi_value: float
    created_at: datetime

    class Config:
        from_attributes = True

# Request/Response schemas for API endpoints
class VIAnalysisRequest(BaseModel):
    field_id: UUID
    vi_type: str
    date_range: Optional[tuple[datetime, datetime]] = None

class VIOverlayRequest(BaseModel):
    geometry: Dict[str, Any]  # GeoJSON
    vi_type: str
    date: Optional[datetime] = None
    date_range: Optional[tuple[datetime, datetime]] = None

class VIOverlayResponse(BaseModel):
    overlay_url: str
    mean_value: float
    min_value: float
    max_value: float
    analysis_message: str

class GeocodeResponse(BaseModel):
    address: str
    lat: float
    lng: float

class SearchLocationRequest(BaseModel):
    query: str
    
class SearchLocationResponse(BaseModel):
    results: List[Dict[str, Any]]
