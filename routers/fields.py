from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse, Response
import tempfile
import os
import io
import zipfile
from sqlalchemy.orm import Session
from database import get_db
from models import User, Field, FieldThumbnail
from schemas import FieldCreate, FieldUpdate, FieldResponse, ThumbnailCreate, ThumbnailResponse
from auth import get_current_user
from geocoding_service import geocoding_service
import json
from shapely.geometry import shape
from shapely.ops import transform
import pyproj
from uuid import UUID

router = APIRouter(prefix="/fields", tags=["fields"])

def calculate_area_and_centroid(geometry: dict):
    """Calculate area in square meters and centroid from GeoJSON geometry"""
    try:
        # Convert to shapely geometry
        geom = shape(geometry)
        
        # Transform to a projected coordinate system (UTM zone 47N for Thailand)
        # to calculate accurate area in square meters
        project = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32647', always_xy=True).transform
        projected_geom = transform(project, geom)
        
        # Calculate area in square meters
        area_m2 = projected_geom.area
        
        # Get centroid in WGS84
        centroid = geom.centroid
        
        return area_m2, centroid.y, centroid.x
    except Exception as e:
        print(f"Error calculating area and centroid: {e}")
        # Return default values if calculation fails
        return 1000.0, 18.78, 98.98

@router.post("/", response_model=FieldResponse)
def create_field(field_data: FieldCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new field"""
    try:
        # Calculate area and centroid
        area_m2, centroid_lat, centroid_lng = calculate_area_and_centroid(field_data.geometry)
        
        # Create new field with default planting_date if not provided
        from datetime import datetime
        planting_date = field_data.planting_date
        if planting_date is None:
            planting_date = datetime.now()  # Use current date as default
            
        # Get address from reverse geocoding
        address = geocoding_service.reverse_geocode_sync(centroid_lat, centroid_lng)
        
        db_field = Field(
            name=field_data.name,
            user_id=current_user.id,
            crop_type=field_data.crop_type,
            variety=field_data.variety,
            planting_season=field_data.planting_season,
            planting_date=planting_date,
            geometry=field_data.geometry,
            area_m2=area_m2,
            centroid_lat=centroid_lat,
            centroid_lng=centroid_lng,
            address=address
        )
        
        db.add(db_field)
        db.commit()
        db.refresh(db_field)
        
        return FieldResponse.model_validate(db_field)
    except Exception as e:
        import traceback
        print(f"Error creating field: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create field: {str(e)}"
        )

@router.get("/", response_model=List[FieldResponse])
def get_user_fields(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all fields for current user"""
    fields = db.query(Field).filter(Field.user_id == current_user.id).all()
    return [FieldResponse.model_validate(field) for field in fields]

@router.get("/{field_id}", response_model=FieldResponse)
def get_field(field_id: UUID, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get a specific field"""
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    return FieldResponse.model_validate(field)

@router.put("/{field_id}", response_model=FieldResponse)
def update_field(
    field_id: UUID,
    field_data: FieldUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a field"""
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    # Update fields
    if field_data.name is not None:
        field.name = field_data.name
    if field_data.crop_type is not None:
        field.crop_type = field_data.crop_type
    if field_data.variety is not None:
        field.variety = field_data.variety
    if field_data.planting_season is not None:
        field.planting_season = field_data.planting_season
    if field_data.planting_date is not None:
        field.planting_date = field_data.planting_date
    
    # Update geometry if provided
    if field_data.geometry is not None:
        area_m2, centroid_lat, centroid_lng = calculate_area_and_centroid(field_data.geometry)
        field.geometry = field_data.geometry
        field.area_m2 = area_m2
        field.centroid_lat = centroid_lat
        field.centroid_lng = centroid_lng
    
    db.commit()
    db.refresh(field)
    
    return FieldResponse.model_validate(field)

@router.delete("/{field_id}")
def delete_field(field_id: UUID, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a field"""
    try:
        field = db.query(Field).filter(
            Field.id == field_id,
            Field.user_id == current_user.id
        ).first()
        
        if not field:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Field not found"
            )
        
        # ลบ thumbnails ที่เกี่ยวข้องก่อน
        from models import FieldThumbnail
        thumbnails = db.query(FieldThumbnail).filter(FieldThumbnail.field_id == field_id).all()
        for thumbnail in thumbnails:
            db.delete(thumbnail)
        
        # ลบ snapshots และ time series ที่เกี่ยวข้อง (ถ้ามี)
        from models import VISnapshot, VITimeSeries
        snapshots = db.query(VISnapshot).filter(VISnapshot.field_id == field_id).all()
        for snapshot in snapshots:
            db.delete(snapshot)
            
        timeseries = db.query(VITimeSeries).filter(VITimeSeries.field_id == field_id).all()
        for ts in timeseries:
            db.delete(ts)
        
        # ลบ field
        db.delete(field)
        db.commit()
        
        return {"message": "Field deleted successfully"}
        
    except Exception as e:
        db.rollback()
        print(f"Error deleting field: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete field: {str(e)}"
        )

@router.post("/{field_id}/thumbnail", response_model=ThumbnailResponse)
def save_field_thumbnail(
    field_id: UUID,
    thumbnail_data: ThumbnailCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save field thumbnail"""
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
    
    # Delete existing thumbnail
    existing_thumbnail = db.query(FieldThumbnail).filter(
        FieldThumbnail.field_id == field_id
    ).first()
    
    if existing_thumbnail:
        db.delete(existing_thumbnail)
    
    # Create new thumbnail
    db_thumbnail = FieldThumbnail(
        field_id=field_id,
        image_data=thumbnail_data.image_data
    )
    
    db.add(db_thumbnail)
    db.commit()
    db.refresh(db_thumbnail)
    
    return ThumbnailResponse.model_validate(db_thumbnail)

@router.get("/{field_id}/thumbnail")
def get_field_thumbnail(
    field_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get field thumbnail"""
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
    
    thumbnail = db.query(FieldThumbnail).filter(
        FieldThumbnail.field_id == field_id
    ).first()
    
    if not thumbnail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thumbnail not found"
        )
    
    return {"image_data": thumbnail.image_data}

@router.post("/import")
async def import_field(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Import field from file (SHP, KML, GeoJSON, GPKG)"""
    # This is a placeholder - full implementation would require 
    # libraries like fiona, geopandas to read different file formats
    try:
        content = await file.read()
        
        # For now, assume it's GeoJSON
        if file.filename.endswith('.geojson'):
            geojson_data = json.loads(content.decode('utf-8'))
            
            # Extract geometry from GeoJSON
            if 'features' in geojson_data and len(geojson_data['features']) > 0:
                geometry = geojson_data['features'][0]['geometry']
                
                # Create field with imported geometry
                field_data = FieldCreate(
                    name="แปลงนำเข้า",
                    geometry=geometry
                )
                
                return create_field(field_data, current_user, db)
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format"
        )
        
    except Exception as e:
        print(f"Error importing field: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import field"
        )

@router.get("/{field_id}/export/{format}")
def export_field(
    field_id: UUID,
    format: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export field in specified format (geojson, kml, csv, shp, gpkg)

    - geojson: downloads a .geojson attachment
    - kml: downloads a .kml attachment
    - csv: downloads a .csv (UTF-8 BOM) with WKT geometry
    - shp/gpkg: not implemented -> 501
    """
    field = db.query(Field).filter(
        Field.id == field_id,
        Field.user_id == current_user.id
    ).first()
    
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Field not found"
        )
    
    fmt = format.lower()

    # Common feature
    feature = {
        "type": "Feature",
        "geometry": field.geometry,
        "properties": {
            "name": field.name,
            "crop_type": field.crop_type,
            "area_m2": field.area_m2,
            "planting_date": field.planting_date.isoformat() if field.planting_date else None,
        },
    }

    if fmt == 'geojson':
        geojson = {
            "type": "FeatureCollection",
            "features": [feature],
        }
        content = json.dumps(geojson, ensure_ascii=False, indent=2)
        headers = {
            "Content-Disposition": f"attachment; filename=field_{field.id}.geojson"
        }
        return Response(content=content, media_type="application/geo+json; charset=utf-8", headers=headers)

    if fmt == 'kml':
        try:
            geom = feature["geometry"]
            def coord_pairs(coords):
                return " ".join([f"{lng},{lat},0" for lng, lat in coords])

            polygons = []
            if geom["type"] == "Polygon":
                polygons = [geom["coordinates"]]
            elif geom["type"] == "MultiPolygon":
                polygons = geom["coordinates"]

            placemarks = []
            for idx, rings in enumerate(polygons):
                outer = rings[0] if rings else []
                coords = coord_pairs(outer)
                name_suffix = f" {idx+1}" if len(polygons) > 1 else ""
                placemarks.append(
                    f"\n    <Placemark>\n      <name>{field.name}{name_suffix}</name>\n      <ExtendedData>\n        <Data name=\"crop_type\"><value>{field.crop_type or ''}</value></Data>\n        <Data name=\"area_m2\"><value>{field.area_m2}</value></Data>\n        <Data name=\"planting_date\"><value>{feature['properties']['planting_date'] or ''}</value></Data>\n      </ExtendedData>\n      <Style><LineStyle><color>ff2b7a4b</color><width>2</width></LineStyle><PolyStyle><color>1a2b7a4b</color></PolyStyle></Style>\n      <Polygon>\n        <outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs>\n      </Polygon>\n    </Placemark>"
                )

            kml = (
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n"
                "  <Document>\n"
                f"    <name>{field.name}</name>" + "".join(placemarks) + "\n"
                "  </Document>\n"
                "</kml>"
            )
            headers = {
                "Content-Disposition": f"attachment; filename=field_{field.id}.kml"
            }
            return Response(content=kml, media_type="application/vnd.google-earth.kml+xml; charset=utf-8", headers=headers)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to export KML: {e}")

    if fmt == 'csv':
        try:
            from shapely.geometry import shape as shp_shape
            geom = shp_shape(feature["geometry"])  # to shapely geometry
            wkt = geom.wkt
            headers_row = ["name","crop_type","area_m2","planting_date","wkt"]
            row = [
                feature["properties"]["name"] or "",
                feature["properties"].get("crop_type") or "",
                str(feature["properties"].get("area_m2") or ""),
                feature["properties"].get("planting_date") or "",
                wkt,
            ]
            def csv_escape(val:str) -> str:
                return '"' + (val.replace('"', '""')) + '"'
            BOM = "\ufeff"
            csv_content = BOM + ",".join(headers_row) + "\n" + ",".join([csv_escape(v) for v in row])
            headers = {
                "Content-Disposition": f"attachment; filename=field_{field.id}.csv"
            }
            return Response(content=csv_content, media_type="text/csv; charset=utf-8", headers=headers)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to export CSV: {e}")

    if fmt in {"shp", "gpkg"}:
        try:
            try:
                import geopandas as gpd  # type: ignore
            except Exception as e:
                return JSONResponse(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    content={"detail": f"GeoPandas/GDAL not available on server: {e}"}
                )

            # Create GeoDataFrame
            geojson = {"type": "FeatureCollection", "features": [feature]}
            gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")

            with tempfile.TemporaryDirectory() as tmpdir:
                if fmt == "gpkg":
                    # Safe ascii filename for gpkg
                    base = (field.name or "field").encode("ascii", "ignore").decode() or f"field_{str(field.id)[:8]}"
                    base = base.replace(" ", "_")
                    out_path = os.path.join(tmpdir, f"{base}.gpkg")
                    gdf.to_file(out_path, driver="GPKG", layer="field")
                    with open(out_path, "rb") as f:
                        data = f.read()
                    headers = {"Content-Disposition": f"attachment; filename={os.path.basename(out_path)}"}
                    return Response(content=data, media_type="application/geopackage+sqlite3", headers=headers)

                # Shapefile -> multiple files, zip them
                shp_dir = os.path.join(tmpdir, "shp")
                os.makedirs(shp_dir, exist_ok=True)
                # Shapefile base name must be <=10 ascii characters
                base_ascii = (field.name or "field").encode("ascii", "ignore").decode()
                if not base_ascii:
                    base_ascii = f"field_{str(field.id)[:2]}"
                base_ascii = base_ascii.replace(" ", "_").lower()
                if len(base_ascii) > 10:
                    base_ascii = base_ascii[:10]
                shp_path = os.path.join(shp_dir, f"{base_ascii}.shp")
                gdf.to_file(shp_path, driver="ESRI Shapefile")

                mem_file = io.BytesIO()
                with zipfile.ZipFile(mem_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fn in os.listdir(shp_dir):
                        full = os.path.join(shp_dir, fn)
                        zf.write(full, arcname=fn)
                mem_file.seek(0)
                headers = {"Content-Disposition": f"attachment; filename={base_ascii}.zip"}
                return Response(content=mem_file.read(), media_type="application/zip", headers=headers)
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Failed to export {fmt}: {e}"}
            )

    raise HTTPException(status_code=400, detail="Unsupported format")
