from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import User, Field, VISnapshot, VITimeSeries
from schemas import (
    VIAnalysisRequest, VIOverlayRequest, VIOverlayResponse,
    VISnapshotCreate, VISnapshotResponse,
    VITimeSeriesCreate, VITimeSeriesResponse
)
from auth import get_current_user
from gee_service import gee_service
from models import VILegendCache
import json
from uuid import UUID

router = APIRouter(prefix="/vi-analysis", tags=["vegetation-indices"])

# Create a second router for backward compatibility with /vi prefix
vi_router = APIRouter(prefix="/vi", tags=["vegetation-indices-compat"])

# Add the timeseries endpoint to the /vi router for compatibility
@vi_router.get("/timeseries/{field_id}")
async def get_vi_timeseries_compat(
    field_id: UUID,
    vi_type: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    analysis_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get VI timeseries data for a field (compatibility endpoint)"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=90)  # Last 3 months
    
    try:
        # Try to get from database first
        db_timeseries = db.query(VITimeSeries).filter(
            VITimeSeries.field_id == field_id,
            VITimeSeries.vi_type == vi_type,
            VITimeSeries.measurement_date.between(start_date, end_date)
        ).order_by(VITimeSeries.measurement_date.asc()).all()
        
        if db_timeseries:
            return {
                "timeseries": [VITimeSeriesResponse.model_validate(ts) for ts in db_timeseries],
                "source": "database"
            }
        
        # If no data in database, try Google Earth Engine
        print(f"🔍 Fetching {vi_type} data from GEE for field {field_id}")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"📊 Analysis type: {analysis_type}")
        print(f"🗺️ Field geometry: {type(field.geometry)} - {len(str(field.geometry))} chars")
        
        # Test GEE connection first
        try:
            import ee
            print("🛰️ Testing GEE connection...")
            test_info = ee.String('Hello World').getInfo()
            print(f"✅ GEE connection successful: {test_info}")
        except Exception as gee_error:
            print(f"❌ GEE connection failed: {gee_error}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Google Earth Engine service unavailable: {str(gee_error)}"
            )
        
        gee_timeseries = gee_service.get_timeseries_data(
            field.geometry, 
            vi_type, 
            start_date, 
            end_date,
            analysis_type
        )
        
        # Save to database for future use
        for datapoint in gee_timeseries:
            measurement_date = datetime.fromisoformat(datapoint['date'].replace('Z', '+00:00'))
            
            # Check if this entry already exists
            existing = db.query(VITimeSeries).filter(
                VITimeSeries.field_id == field_id,
                VITimeSeries.vi_type == vi_type,
                VITimeSeries.measurement_date == measurement_date
            ).first()
            
            if not existing:
                timeseries_entry = VITimeSeries(
                    field_id=field_id,
                    vi_type=vi_type,
                    measurement_date=measurement_date,
                    vi_value=datapoint['value']
                )
                db.add(timeseries_entry)
        
        db.commit()
        
        # Return the data in the correct format
        formatted_timeseries = []
        for d in gee_timeseries:
            formatted_timeseries.append({
                "measurement_date": d['date'],
                "vi_value": d['value']
            })
        
        print(f"✅ Successfully fetched {len(formatted_timeseries)} data points from GEE")
        
        return {
            "timeseries": formatted_timeseries,
            "source": "google_earth_engine",
            "analysis_type": analysis_type,
            "count": len(formatted_timeseries)
        }
        
    except Exception as e:
        print(f"Error getting timeseries data: {e}")
        # If both database and GEE fail, return empty data
        return {
            "timeseries": [],
            "source": "no_data_available",
            "message": "No timeseries data available. Please check GEE configuration."
        }

# Add snapshots endpoint too for compatibility
@vi_router.get("/snapshots/{field_id}")
async def get_vi_snapshots_compat(
    field_id: UUID,
    vi_type: Optional[str] = "NDVI",
    limit: Optional[int] = 4,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get VI snapshots for a field (compatibility endpoint)"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    # Get snapshots from database
    snapshots = db.query(VISnapshot).filter(
        VISnapshot.field_id == field_id,
        VISnapshot.vi_type == vi_type
    ).order_by(VISnapshot.snapshot_date.desc()).limit(limit).all()
    
    # If no snapshots or need fresh data, try to get latest images from GEE
    if len(snapshots) < limit:
        try:
            print(f"Getting latest images for field {field_id} from GEE...")
            
            # Get latest 4 images from GEE
            latest_images = gee_service.get_latest_images_data(
                geometry=field.geometry,
                vi_type=vi_type,
                limit=limit
            )
            
            new_snapshots = []
            for image_data in latest_images:
                # Check if snapshot for this date already exists
                acquisition_date = datetime.fromisoformat(image_data['acquisition_date'].replace('Z', ''))
                existing = db.query(VISnapshot).filter(
                    VISnapshot.field_id == field_id,
                    VISnapshot.vi_type == vi_type,
                    VISnapshot.snapshot_date.date() == acquisition_date.date()
                ).first()
                
                if not existing:
                    # Save new snapshot to database
                    snapshot = VISnapshot(
                        field_id=field_id,
                        user_id=field.user_id,
                        vi_type=vi_type,
                        snapshot_date=acquisition_date,
                        mean_value=image_data['mean_value'],
                        min_value=image_data['min_value'],
                        max_value=image_data['max_value'],
                        overlay_data=image_data['overlay_url'],
                        status_message=image_data['analysis_message']
                    )
                    
                    db.add(snapshot)
                    new_snapshots.append(snapshot)
            
            if new_snapshots:
                db.commit()
                for snapshot in new_snapshots:
                    db.refresh(snapshot)
                print(f"✅ Saved {len(new_snapshots)} new snapshots")
            
            # Get updated snapshots from database
            updated_snapshots = db.query(VISnapshot).filter(
                VISnapshot.field_id == field_id,
                VISnapshot.vi_type == vi_type
            ).order_by(VISnapshot.snapshot_date.desc()).limit(limit).all()
            
            if updated_snapshots:
                return [VISnapshotResponse.model_validate(snapshot) for snapshot in updated_snapshots]
            
        except Exception as e:
            print(f"Failed to generate real snapshot: {e}")
            
            # If GEE fails, return empty list - only real data allowed
            return []
    
    return [VISnapshotResponse.model_validate(snapshot) for snapshot in snapshots]



@router.post("/overlay", response_model=VIOverlayResponse)
def generate_vi_overlay(
    request: VIOverlayRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate VI overlay for given geometry and parameters"""
    try:
        # Use current date if not specified
        analysis_date = request.date or datetime.now()
        
        # Get VI statistics from GEE
        stats = gee_service.get_vi_statistics(
            geometry=request.geometry,
            vi_type=request.vi_type,
            date=analysis_date
        )
        
        # Generate overlay image
        overlay_url = gee_service.generate_vi_overlay(
            geometry=request.geometry,
            vi_type=request.vi_type,
            date=analysis_date
        )
        
        return VIOverlayResponse(
            overlay_url=overlay_url,
            mean_value=stats['mean_value'],
            min_value=stats['min_value'],
            max_value=stats['max_value'],
            analysis_message=stats['analysis_message']
        )
        
    except Exception as e:
        print(f"Error generating VI overlay: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate VI overlay"
        )

@router.delete("/snapshots/{field_id}")
def delete_field_snapshots(
    field_id: UUID,
    vi_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete VI snapshots for a field"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    try:
        # Build query to delete snapshots
        query = db.query(VISnapshot).filter(VISnapshot.field_id == field_id)
        
        if vi_type:
            query = query.filter(VISnapshot.vi_type == vi_type)
            print(f"🗑️ Deleting {vi_type} snapshots for field {field_id}")
        else:
            print(f"🗑️ Deleting ALL snapshots for field {field_id}")
        
        # Get snapshots to delete (for counting)
        snapshots_to_delete = query.all()
        count_to_delete = len(snapshots_to_delete)
        
        # Delete snapshots
        deleted_count = query.delete()
        db.commit()
        
        print(f"✅ Deleted {deleted_count} snapshots")
        
        return {
            "message": f"Deleted {deleted_count} snapshots successfully",
            "deleted_count": deleted_count,
            "vi_type": vi_type or "ALL",
            "field_id": str(field_id)
        }
        
    except Exception as e:
        print(f"❌ Error deleting snapshots: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete snapshots"
        )

@router.post("/{field_id}/analyze-historical")
def analyze_historical_vi(
    field_id: UUID,
    vi_type: str = "NDVI",
    count: int = 4,
    clear_old: bool = True,  # New parameter to clear old snapshots
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate multiple historical VI snapshots for a field"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    try:
        print(f"🔍 Generating {count} historical {vi_type} snapshots for field {field_id}")
        
        # Clear old snapshots of the same VI type if requested
        if clear_old:
            print(f"🗑️ Clearing old {vi_type} snapshots...")
            old_snapshots = db.query(VISnapshot).filter(
                VISnapshot.field_id == field_id,
                VISnapshot.vi_type == vi_type
            )
            old_count = old_snapshots.count()
            if old_count > 0:
                old_snapshots.delete()
                db.commit()
                print(f"✅ Cleared {old_count} old snapshots")
        
        # Use the new get_latest_images_data function for diverse, clean historical data
        historical_images = gee_service.get_latest_images_data(
            geometry=field.geometry,
            vi_type=vi_type,
            limit=count
        )
        
        if not historical_images:
            print("⚠️ No historical images available, cannot create snapshots")
            return {
                "message": "No historical images available for this field",
                "snapshots_created": 0,
                "vi_type": vi_type,
                "field_id": str(field_id)
            }
        
        snapshots_created = []
        
        for i, image_data in enumerate(historical_images):
            try:
                # Parse the acquisition date
                analysis_date = datetime.fromisoformat(image_data['acquisition_date'].replace('Z', '+00:00'))
                
                print(f"📅 Processing snapshot {i+1}/{len(historical_images)} for date: {analysis_date.strftime('%Y-%m-%d')}")
                
                # Check if snapshot already exists for this date (within same day)
                from sqlalchemy import func, Date
                existing = db.query(VISnapshot).filter(
                    VISnapshot.field_id == field_id,
                    VISnapshot.vi_type == vi_type,
                    func.date(VISnapshot.snapshot_date) == analysis_date.date()
                ).first()
                
                if existing:
                    print(f"⚠️ Snapshot already exists for {analysis_date.date()}, skipping")
                    continue
                
                # Save snapshot to database using data from get_latest_images_data
                snapshot = VISnapshot(
                    field_id=field_id,
                    user_id=current_user.id,
                    vi_type=vi_type,
                    snapshot_date=analysis_date,
                    mean_value=image_data['mean_value'],
                    min_value=image_data['min_value'],
                    max_value=image_data['max_value'],
                    overlay_data=image_data['overlay_url'],
                    status_message=image_data['analysis_message']
                )
                
                db.add(snapshot)
                snapshots_created.append(snapshot)
                
            except Exception as e:
                print(f"❌ Failed to create snapshot {i+1}: {e}")
                continue
        
        # Commit all snapshots at once
        db.commit()
        
        # Refresh all snapshots
        for snapshot in snapshots_created:
            db.refresh(snapshot)
        
        print(f"✅ Successfully created {len(snapshots_created)} historical snapshots from {len(historical_images)} images")
        
        return {
            "message": f"Historical analysis completed for {len(snapshots_created)} snapshots with unique dates",
            "snapshots_created": len(snapshots_created),
            "vi_type": vi_type,
            "field_id": str(field_id),
            "unique_dates": len(set(s.snapshot_date.date() for s in snapshots_created))
        }
        
    except Exception as e:
        print(f"❌ Historical analysis failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate historical analysis: {str(e)}"
        )

@router.post("/{field_id}/analyze")
def analyze_and_save_vi(
    field_id: UUID,
    vi_type: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze VI for a field and save snapshot"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    try:
        analysis_date = datetime.now()
        
        # Get VI statistics from GEE
        stats = gee_service.get_vi_statistics(
            geometry=field.geometry,
            vi_type=vi_type,
            date=analysis_date
        )
        
        # Generate overlay
        overlay_url = gee_service.generate_vi_overlay(
            geometry=field.geometry,
            vi_type=vi_type,
            date=analysis_date
        )
        
        # Save snapshot to database
        snapshot = VISnapshot(
            field_id=field_id,
            user_id=current_user.id,
            vi_type=vi_type,
            snapshot_date=analysis_date,
            mean_value=stats['mean_value'],
            min_value=stats['min_value'],
            max_value=stats['max_value'],
            overlay_data=overlay_url,
            status_message=stats['analysis_message']
        )
        
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        
        return {
            "message": "VI analysis completed and saved",
            "snapshot_id": str(snapshot.id),
            "mean_value": stats['mean_value'],
            "analysis_message": stats['analysis_message']
        }
        
    except Exception as e:
        print(f"Error analyzing and saving VI: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze and save VI"
        )

@router.get("/{field_id}/current")
def get_current_vi_analysis(
    field_id: UUID,
    vi_type: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current VI analysis for a field"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    try:
        analysis_date = datetime.now()
        
        # Get VI statistics from GEE
        stats = gee_service.get_vi_statistics(
            geometry=field.geometry,
            vi_type=vi_type,
            date=analysis_date
        )
        
        # Generate overlay
        overlay_url = gee_service.generate_vi_overlay(
            geometry=field.geometry,
            vi_type=vi_type,
            date=analysis_date
        )
        
        return {
            "field_id": str(field_id),
            "vi_type": vi_type,
            "analysis_date": analysis_date.isoformat(),
            "mean_value": stats['mean_value'],
            "min_value": stats['min_value'],
            "max_value": stats['max_value'],
            "analysis_message": stats['analysis_message'],
            "overlay_data": overlay_url
        }
        
    except Exception as e:
        print(f"Error getting VI analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get VI analysis"
        )

@router.get("/snapshots/{field_id}")
def get_field_snapshots(
    field_id: UUID,
    vi_type: Optional[str] = None,
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get VI snapshots for a field"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    query = db.query(VISnapshot).filter(VISnapshot.field_id == field_id)
    
    if vi_type:
        query = query.filter(VISnapshot.vi_type == vi_type)
    
    snapshots = query.order_by(VISnapshot.snapshot_date.desc()).limit(limit).all()
    
    # If no snapshots found, try to create real snapshots from GEE
    if not snapshots:
        try:
            print(f"No snapshots found for field {field_id}, attempting to generate from GEE...")
            
            # Get the field to access its geometry
            field_for_analysis = db.query(Field).filter(Field.id == field_id).first()
            if not field_for_analysis:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Field not found for analysis"
                )
            
            # Try to create a real analysis using GEE
            analysis_date = datetime.now()
            vi_type_to_use = vi_type or "NDVI"
            
            # Get VI statistics from GEE
            stats = gee_service.get_vi_statistics(
                geometry=field_for_analysis.geometry,
                vi_type=vi_type_to_use,
                date=analysis_date
            )
            
            # Generate overlay
            overlay_url = gee_service.generate_vi_overlay(
                geometry=field_for_analysis.geometry,
                vi_type=vi_type_to_use,
                date=analysis_date
            )
            
            # Save snapshot to database
            snapshot = VISnapshot(
                field_id=field_id,
                user_id=field_for_analysis.user_id,
                vi_type=vi_type_to_use,
                snapshot_date=analysis_date,
                mean_value=stats['mean_value'],
                min_value=stats['min_value'],
                max_value=stats['max_value'],
                overlay_data=overlay_url,
                status_message=stats['analysis_message']
            )
            
            db.add(snapshot)
            db.commit()
            db.refresh(snapshot)
            
            print(f"✅ Generated real snapshot: {vi_type_to_use} = {stats['mean_value']:.3f}")
            
            return [VISnapshotResponse.model_validate(snapshot)]
            
        except Exception as e:
            print(f"Failed to generate real snapshot: {e}")
            # If GEE fails completely, return empty list
            return []
    
    return [VISnapshotResponse.model_validate(snapshot) for snapshot in snapshots]



@router.get("/timeseries/{field_id}")
def get_field_timeseries(
    field_id: UUID,
    vi_type: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    analysis_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get VI timeseries data for a field"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=90)  # Last 3 months
    
    try:
        # Try to get from database first
        db_timeseries = db.query(VITimeSeries).filter(
            VITimeSeries.field_id == field_id,
            VITimeSeries.vi_type == vi_type,
            VITimeSeries.measurement_date.between(start_date, end_date)
        ).order_by(VITimeSeries.measurement_date.asc()).all()
        
        # Check if we have sufficient data in DB for the requested period
        if db_timeseries and len(db_timeseries) >= 1:
            print(f"✅ Found {len(db_timeseries)} existing records in database")
            return {
                "timeseries": [VITimeSeriesResponse.model_validate(ts) for ts in db_timeseries],
                "source": "database",
                "count": len(db_timeseries)
            }
        
        # Otherwise, get fresh data from GEE
        print(f"🔍 Fetching {vi_type} data from GEE for field {field_id}")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"📊 Analysis type: {analysis_type}")
        print(f"🗺️ Field geometry: {type(field.geometry)} - {len(str(field.geometry))} chars")
        
        # Test GEE connection first
        try:
            import ee
            print("🛰️ Testing GEE connection...")
            test_info = ee.String('Hello World').getInfo()
            print(f"✅ GEE connection successful: {test_info}")
        except Exception as gee_error:
            print(f"❌ GEE connection failed: {gee_error}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Google Earth Engine service unavailable: {str(gee_error)}"
            )
        
        gee_timeseries = gee_service.get_timeseries_data(
            geometry=field.geometry,
            vi_type=vi_type,
            start_date=start_date,
            end_date=end_date,
            analysis_type=analysis_type
        )
        
        # Save new data to database (monthly averages)
        saved_count = 0
        for data_point in gee_timeseries:
            try:
                # Parse date - data_point['date'] is already ISO format from GEE
                if 'T' in data_point['date']:
                    measurement_date = datetime.fromisoformat(data_point['date'].replace('Z', '+00:00'))
                else:
                    measurement_date = datetime.fromisoformat(data_point['date'])
                
                # Check if entry already exists for this month
                existing = db.query(VITimeSeries).filter(
                    VITimeSeries.field_id == field_id,
                    VITimeSeries.vi_type == vi_type,
                    VITimeSeries.measurement_date >= measurement_date.replace(day=1),
                    VITimeSeries.measurement_date < (measurement_date.replace(day=28) + timedelta(days=4)).replace(day=1)
                ).first()
                
                if not existing:
                    timeseries_entry = VITimeSeries(
                        field_id=field_id,
                        vi_type=vi_type,
                        measurement_date=measurement_date,
                        vi_value=data_point['value']
                    )
                    db.add(timeseries_entry)
                    saved_count += 1
                    print(f"   💾 Saved: {measurement_date.strftime('%B %Y')} = {data_point['value']:.3f}")
                else:
                    print(f"   ⏭️ Skipped: {measurement_date.strftime('%B %Y')} (already exists)")
                    
            except Exception as e:
                print(f"   ❌ Error saving data point: {e}")
                continue
        
        db.commit()
        print(f"💾 Saved {saved_count} new records to database")
        
        # Return the data in the correct format
        formatted_timeseries = []
        for d in gee_timeseries:
            formatted_timeseries.append({
                "measurement_date": d['date'],
                "vi_value": d['value']
            })
        
        print(f"✅ Successfully fetched {len(formatted_timeseries)} monthly averages from GEE")
        
        return {
            "timeseries": formatted_timeseries,
            "source": "google_earth_engine",
            "analysis_type": analysis_type,
            "count": len(formatted_timeseries),
            "saved_to_db": saved_count
        }
        
    except Exception as e:
        print(f"❌ Error getting timeseries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get timeseries data: {str(e)}"
        )

# ลบ available-dates endpoint เพราะไม่จำเป็น - frontend จะใช้ข้อมูลแบบ hardcode

@router.get("/latest/{field_id}")  
def get_latest_vi_values(
    field_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get latest VI values for all indices for a field"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    vi_types = ['NDVI', 'EVI', 'SAVI', 'GNDVI', 'NDRE', 'LSWI']
    latest_values = {}
    
    for vi_type in vi_types:
        latest_snapshot = db.query(VISnapshot).filter(
            VISnapshot.field_id == field_id,
            VISnapshot.vi_type == vi_type
        ).order_by(VISnapshot.snapshot_date.desc()).first()
        
        if latest_snapshot:
            latest_values[vi_type] = {
                "value": latest_snapshot.mean_value,
                "date": latest_snapshot.snapshot_date.isoformat(),
                "analysis_message": latest_snapshot.analysis_message
            }
        else:
            latest_values[vi_type] = None
    
    return latest_values

@router.post("/bulk-analyze/{field_id}")
def bulk_analyze_field(
    field_id: UUID,
    vi_types: List[str],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze multiple VI types for a field at once"""
    # Verify field ownership
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    results = {}
    analysis_date = datetime.now()
    
    for vi_type in vi_types:
        try:
            # Get VI statistics
            stats = gee_service.get_vi_statistics(
                geometry=field.geometry,
                vi_type=vi_type,
                date=analysis_date
            )
            
            # Generate overlay
            overlay_url = gee_service.generate_vi_overlay(
                geometry=field.geometry,
                vi_type=vi_type,
                date=analysis_date
            )
            
            # Save snapshot
            snapshot = VISnapshot(
                field_id=field_id,
                vi_type=vi_type,
                snapshot_date=analysis_date,
                mean_value=stats['mean_value'],
                min_value=stats['min_value'],
                max_value=stats['max_value'],
                overlay_url=overlay_url,
                analysis_message=stats['analysis_message']
            )
            
            db.add(snapshot)
            
            # Save timeseries entry
            timeseries_entry = VITimeSeries(
                field_id=field_id,
                vi_type=vi_type,
                measurement_date=analysis_date,
                vi_value=stats['mean_value']
            )
            
            db.add(timeseries_entry)
            
            results[vi_type] = {
                "success": True,
                "stats": stats,
                "overlay_url": overlay_url
            }
            
        except Exception as e:
            print(f"Error analyzing {vi_type}: {e}")
            results[vi_type] = {
                "success": False,
                "error": str(e)
            }
    
    db.commit()
    
    return {
        "message": "Bulk analysis completed",
        "results": results,
        "analysis_date": analysis_date.isoformat()
    }

# New combined tiles + legend endpoint per contract
@vi_router.get("/tiles")
def get_vi_tiles(
    vi: str,
    start: str,
    end: str,
    aoi_geojson: str,
    mode: str = 'static',
    current_user: User = Depends(get_current_user)
):
    """Return XYZ tiles URL and discrete legend aligned with visualization.
    Query params:
    - vi: NDVI,EVI,SAVI,GNDVI,NDRE,LSWI
    - start,end: ISO date strings (YYYY-MM-DD)
    - aoi_geojson: stringified GeoJSON FeatureCollection (first feature is AOI polygon)
    - mode: static|dynamic
    """
    try:
        # Parse AOI
        import json as _json
        fc = _json.loads(aoi_geojson)
        features = fc.get('features') or []
        if not features:
            raise HTTPException(status_code=400, detail="Invalid AOI: empty features")
        geom = features[0].get('geometry')
        if not geom:
            raise HTTPException(status_code=400, detail="Invalid AOI: missing geometry")

        # Cache lookup
        aoi_hash = VILegendCache.compute_aoi_hash(geom)
        from database import SessionLocal
        db = SessionLocal()
        try:
            cached = db.query(VILegendCache).filter(
                VILegendCache.vi == vi,
                VILegendCache.start_date == start,
                VILegendCache.end_date == end,
                VILegendCache.mode == mode,
                VILegendCache.aoi_hash == aoi_hash
            ).first()
            if cached:
                return {
                    "available": True,
                    "tiles": cached.tiles_url,
                    "legend": json.loads(cached.legend_json)
                }
        finally:
            db.close()

        # Delegate to GEE service
        result = gee_service.get_vi_tiles_and_legend(geom, vi, start, end, mode)
        # Map to contract field names
        if not result.get('available'):
            return {"available": False, "reason": result.get('reason', 'insufficient data for selected AOI/time')}
        # Save to cache
        db = SessionLocal()
        try:
            cache_row = VILegendCache(
                vi=vi,
                start_date=start,
                end_date=end,
                mode=mode,
                aoi_hash=aoi_hash,
                tiles_url=result['tiles'],
                legend_json=json.dumps(result['legend'])
            )
            db.add(cache_row)
            db.commit()
        finally:
            db.close()
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"/vi/tiles failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate tiles")