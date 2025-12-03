from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid
import hashlib
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    date_of_birth = Column(DateTime, nullable=True)
    age = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    fields = relationship("Field", back_populates="owner")

class Field(Base):
    __tablename__ = "fields"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, default="แปลง A1")
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    crop_type = Column(String, nullable=True, default="ข้าวหอมมะลิ")
    variety = Column(String, nullable=True, default="ข้าวหอมมะลิ")  # สายพันธุ์ข้าว
    planting_season = Column(String, nullable=True)  # ฤดูกาลปลูก
    planting_date = Column(DateTime, nullable=True)
    geometry = Column(JSONB, nullable=False)  # GeoJSON polygon
    area_m2 = Column(Float, nullable=False)
    centroid_lat = Column(Float, nullable=False)
    centroid_lng = Column(Float, nullable=False)
    address = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="fields")
    thumbnails = relationship("FieldThumbnail", back_populates="field")
    snapshots = relationship("VISnapshot", back_populates="field")
    timeseries = relationship("VITimeSeries", back_populates="field")

class FieldThumbnail(Base):
    __tablename__ = "thumbnails"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id"), nullable=False)
    image_data = Column(Text, nullable=False)  # Base64 data URL
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    field = relationship("Field", back_populates="thumbnails")

class VISnapshot(Base):
    __tablename__ = "snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    vi_type = Column(String(10), nullable=False)  # NDVI, EVI, GNDVI, NDWI, SAVI, VCI
    snapshot_date = Column(DateTime, nullable=False)
    mean_value = Column(Float, nullable=False)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    overlay_data = Column(Text, nullable=True)  # File path to VI overlay image
    grid_data = Column(JSONB, nullable=True)  # Optional: store full grid
    status_message = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=True)
    
    # Relationships
    field = relationship("Field", back_populates="snapshots")

class VITimeSeries(Base):
    __tablename__ = "time_series"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    field_id = Column(UUID(as_uuid=True), ForeignKey("fields.id"), nullable=False)
    vi_type = Column(String, nullable=False)
    measurement_date = Column(DateTime, nullable=False)
    vi_value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    field = relationship("Field", back_populates="timeseries")

class ImportExportLog(Base):
    __tablename__ = "import_export"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    action_type = Column(String, nullable=False)  # 'import' or 'export'
    file_type = Column(String, nullable=False)  # 'shp', 'kml', 'geojson', 'gpkg'
    file_name = Column(String, nullable=False)
    status = Column(String, nullable=False)  # 'success' or 'failed'
    created_at = Column(DateTime, default=datetime.utcnow)


class VILegendCache(Base):
    __tablename__ = "vi_legend_cache"
    id = Column(Integer, primary_key=True, index=True)
    vi = Column(String, index=True)
    start_date = Column(String, index=True)
    end_date = Column(String, index=True)
    mode = Column(String, default='static', index=True)
    aoi_hash = Column(String, index=True)
    tiles_url = Column(Text)
    legend_json = Column(Text)  # store as JSON string for portability
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    @staticmethod
    def compute_aoi_hash(geometry_dict: dict) -> str:
        m = hashlib.sha256()
        m.update(str(geometry_dict).encode('utf-8'))
        return m.hexdigest()[:40]
