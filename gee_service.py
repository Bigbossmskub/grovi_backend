import ee
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from config import settings
from image_service import image_service
import os

class GEEService:
    def __init__(self):
        """Initialize Google Earth Engine service"""
        self.initialize_gee()
        
    def initialize_gee(self):
        """Initialize GEE with service account"""
        self.gee_available = False
        try:
            # Use service account credentials - find key file intelligently
            key_file_path = self._find_key_file()
            
            if not key_file_path:
                print("❌ GEE key file not found in any expected location")
                print("⚠️ กรุณาติดต่อผู้ดูแลระบบเพื่อตั้งค่า Google Earth Engine")
                return
                
            credentials = ee.ServiceAccountCredentials(
                settings.gee_service_account_email,
                key_file_path
            )
            ee.Initialize(credentials, project=settings.gee_project_id)
            
            # Test if GEE is actually working by making a simple call
            ee.Number(1).getInfo()
            self.gee_available = True
            print("✅ Google Earth Engine initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize GEE: {e}")
            if "Not signed up" in str(e) or "project is not registered" in str(e):
                print("⚠️ GEE Project ไม่ได้ลงทะเบียนหรือ Service Account ไม่มีสิทธิ์")
                print("💡 กรุณาตรวจสอบ:")
                print("   1. Project ID ถูกต้องหรือไม่")
                print("   2. Service Account มีสิทธิ์ใน Earth Engine หรือไม่")
                print("   3. ลงทะเบียน Earth Engine Project แล้วหรือยัง")
            self.gee_available = False
    
    def _find_key_file(self):
        """Find GEE key file in multiple possible locations"""
        key_filename = settings.gee_key_file
        
        # List of possible locations to search for the key file
        search_paths = [
            # 1. Current working directory
            key_filename,
            # 2. Backend directory
            os.path.join(os.path.dirname(__file__), key_filename),
            # 3. Parent directory (project root)
            os.path.join(os.path.dirname(__file__), '..', key_filename),
            # 4. Project root (relative from backend)
            os.path.join(os.path.dirname(__file__), '..', '..', key_filename),
            # 5. Absolute path if provided
            os.path.abspath(key_filename),
            # 6. Hardcoded project paths
            f"D:\\1683.8.68\\{key_filename}",
            f"D:\\1683.8.68\\backend\\{key_filename}",
            # 7. Try common locations
            os.path.expanduser(f"~/{key_filename}"),
            f"./{key_filename}",
            f"../{key_filename}"
        ]
        
        print(f"🔍 Searching for GEE key file: {key_filename}")
        
        for i, path in enumerate(search_paths, 1):
            try:
                abs_path = os.path.abspath(path)
                print(f"   {i}. Checking: {abs_path}")
                
                if os.path.exists(abs_path) and os.path.isfile(abs_path):
                    file_size = os.path.getsize(abs_path)
                    if file_size > 0:
                        print(f"   ✅ Found valid key file at: {abs_path} ({file_size} bytes)")
                        return abs_path
                    else:
                        print(f"   ⚠️  Key file exists but is empty: {abs_path}")
            except Exception as e:
                print(f"   ❌ Error checking path {path}: {e}")
        
        print(f"   ❌ Valid key file not found in any location")
        print(f"   💡 Please ensure '{key_filename}' exists in project root or backend folder")
        return None
    
    def get_sentinel2_collection(self, geometry: Dict, start_date: str, end_date: str):
        """Get Sentinel-2 collection for specified geometry and date range with cloud masking"""
        try:
            # Convert GeoJSON to EE geometry
            if not geometry:
                raise ValueError("Geometry is required but not provided")
            
            ee_geometry = ee.Geometry(geometry)
            
            # Validate geometry bounds (make sure it's reasonable)
            bounds = ee_geometry.bounds().getInfo()
            if not bounds:
                raise ValueError("Invalid geometry bounds")
            
            # Get Sentinel-2 collection with stricter cloud filtering for better quality
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(ee_geometry)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))  # Stricter cloud filter
                         .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'QA60']))  # Include B5 for NDRE and QA60 for cloud masking
            
            # Apply cloud masking function
            def apply_cloud_mask(image):
                """Apply cloud and cirrus masking using QA60 band"""
                qa = image.select('QA60')
                
                # Ensure QA60 is treated as integer for bitwise operations
                qa = qa.toInt()
                
                # Cloud mask (bit 10) and cirrus mask (bit 11)
                cloud_bit_mask = 1 << 10
                cirrus_bit_mask = 1 << 11
                
                # Both flags should be set to zero, indicating clear conditions
                mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                    qa.bitwiseAnd(cirrus_bit_mask).eq(0))
                
                # Return the masked image, scaled to reflectance values
                return image.updateMask(mask).multiply(0.0001).copyProperties(image, ["system:time_start"])
            
            # Apply the cloud mask to all images
            collection = collection.map(apply_cloud_mask)
            
            collection_size = collection.size().getInfo()
            print(f"Found {collection_size} cloud-masked images for date range {start_date} to {end_date}")
            
            # If no images found with strict criteria, try with extended date range only
            if collection_size == 0:
                print(f"No cloud-free images found in 180 days, trying extended 1-year range...")
                
                # Try with wider date range but keep strict quality standards
                from datetime import datetime, timedelta
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                start_dt = end_dt - timedelta(days=365)  # Extended to 1 year
                
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(ee_geometry)
                             .filterDate(start_dt.strftime('%Y-%m-%d'), end_date)
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))  # Keep strict cloud filter
                             .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'QA60']))
                
                # Apply the same strict cloud masking
                collection = collection.map(apply_cloud_mask)
                collection_size = collection.size().getInfo()
                print(f"Found {collection_size} high-quality images in extended 1-year range")
            
            return collection, ee_geometry
        except Exception as e:
            print(f"Error getting Sentinel-2 collection: {e}")
            raise
    
    def calculate_vi(self, image: ee.Image, vi_type: str) -> ee.Image:
        """Calculate vegetation index"""
        try:
            # Check if image is valid
            if image is None:
                raise ValueError("Image is None")
            
            if vi_type == 'NDVI':
                # Use manual calculation instead of normalizedDifference to avoid parameter issues
                nir = image.select('B8')
                red = image.select('B4')
                return nir.subtract(red).divide(nir.add(red)).rename('VI')
            elif vi_type == 'EVI':
                return image.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4'),
                        'BLUE': image.select('B2')
                    }
                ).rename('VI')
            elif vi_type == 'GNDVI':
                # Use manual calculation instead of normalizedDifference
                nir = image.select('B8')
                green = image.select('B3')
                return nir.subtract(green).divide(nir.add(green)).rename('VI')
            elif vi_type == 'NDRE':
                # (NIR - RedEdge) / (NIR + RedEdge); Sentinel-2 RedEdge band B5
                nir = image.select('B8')
                rededge = image.select('B5')
                return nir.subtract(rededge).divide(nir.add(rededge)).rename('VI')
            elif vi_type == 'SAVI':
                return image.expression(
                    '((NIR - RED) / (NIR + RED + 0.5)) * (1 + 0.5)',
                    {
                        'NIR': image.select('B8'),
                        'RED': image.select('B4')
                    }
                ).rename('VI')
            elif vi_type == 'LSWI':
                # (NIR - SWIR) / (NIR + SWIR); Sentinel-2 SWIR band B11
                nir = image.select('B8')
                swir = image.select('B11')
                return nir.subtract(swir).divide(nir.add(swir)).rename('VI')
            else:
                raise ValueError(f"Unknown VI type: {vi_type}")
        except Exception as e:
            print(f"Error calculating VI {vi_type}: {e}")
            raise
    
    def get_vi_statistics(self, geometry: Dict, vi_type: str, date: Optional[datetime] = None) -> Dict:
        """Get VI statistics for a field geometry"""
        
        # Check if GEE is available
        if not self.gee_available:
            raise Exception("ไม่สามารถเชื่อมต่อ Google Earth Engine ได้ กรุณาติดต่อผู้ดูแลระบบ")
        
        try:
            if date is None:
                date = datetime.now()
            
            # Get date range (±7 days from target date for better cloud coverage)
            start_date = (date - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = (date + timedelta(days=7)).strftime('%Y-%m-%d')
            
            collection, ee_geometry = self.get_sentinel2_collection(geometry, start_date, end_date)
            
            # Check if collection is empty
            collection_size = collection.size().getInfo()
            if collection_size == 0:
                raise ValueError("No suitable images found for the specified date range")
            
            # Get the most recent image
            image = collection.sort('system:time_start', False).first()
            
            # Double check that image is not null
            if image is None:
                raise ValueError("Failed to get image from collection")
            
            # Check if image has required bands before calculating VI
            try:
                image_bands = image.bandNames().getInfo()
                required_bands = ['B8', 'B4']  # NIR and RED for most VIs
                
                if vi_type == 'EVI':
                    required_bands = ['B8', 'B4', 'B2']
                elif vi_type in ['GNDVI']:
                    required_bands = ['B8', 'B3']
                elif vi_type == 'NDRE':
                    required_bands = ['B8', 'B5']
                elif vi_type == 'LSWI':
                    required_bands = ['B8', 'B11']
                
                missing_bands = [band for band in required_bands if band not in image_bands]
                if missing_bands:
                    raise ValueError(f"Image missing required bands: {missing_bands}")
                    
            except Exception as e:
                print(f"Error checking image bands: {e}")
                raise ValueError("Image bands validation failed")
            
            # Calculate VI
            vi_image = self.calculate_vi(image, vi_type)
            
            # Verify VI image is valid
            if vi_image is None:
                raise ValueError("VI calculation returned null image")
            
            # Get statistics
            # Use 20m for indices relying on 20m bands (e.g., NDRE uses B5, LSWI uses B11)
            analysis_scale = 20 if vi_type in ['NDRE', 'LSWI'] else 10

            stats = vi_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.minMax(),
                    sharedInputs=True
                ),
                geometry=ee_geometry,
                scale=analysis_scale,
                maxPixels=1e9
            ).getInfo()
            
            # Extract values with fallbacks
            mean_value = stats.get('VI_mean', None)
            min_value = stats.get('VI_min', None)
            max_value = stats.get('VI_max', None)
            
            # Check if we got valid statistics
            if mean_value is None:
                raise ValueError("Failed to calculate VI statistics - no data returned")
            
            # Allow zero values for some VIs (like LSWI which can be negative)
            if vi_type not in ['LSWI'] and mean_value == 0:
                raise ValueError("Failed to calculate VI statistics - zero value detected")
            
            # Generate analysis message
            analysis_message = self.generate_analysis_message(mean_value, vi_type)
            
            return {
                'mean_value': float(mean_value),
                'min_value': float(min_value) if min_value is not None else float(mean_value),
                'max_value': float(max_value) if max_value is not None else float(mean_value),
                'analysis_message': analysis_message,
                'measurement_date': date.isoformat()
            }
            
        except Exception as e:
            print(f"Error getting VI statistics: {e}")
            raise Exception("ไม่สามารถดึงข้อมูลจากดาวเทียมได้ กรุณาลองใหม่อีกครั้ง")
    
    def _get_static_edges_palette(self, vi_type: str) -> Tuple[List[float], List[str]]:
        """Return static edges and palette per VI as per spec."""
        if vi_type == 'NDVI':
            return [-0.2, 0.0, 0.2, 0.4, 0.7, 1.0], ["#a50026","#f46d43","#fee08b","#abdda4","#3288bd"]
        if vi_type in ['EVI', 'SAVI', 'GNDVI']:
            return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["#a50026","#f46d43","#fee08b","#abdda4","#3288bd"]
        if vi_type == 'NDRE':
            return [-0.10, 0.05, 0.20, 0.35, 0.50, 0.60], ["#5e4fa2","#3288bd","#abdda4","#fee08b","#f46d43"]
        if vi_type == 'LSWI':
            return [-1.0, -0.5, -0.1, 0.2, 0.5, 1.0], ["#313695","#74add1","#abd9e9","#fee090","#d73027"]
        # default fallback
        return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["#a50026","#f46d43","#fee08b","#abdda4","#3288bd"]

    def _compute_dynamic_edges(self, vi_image: 'ee.Image', geometry: 'ee.Geometry', vi_type: str) -> List[float]:
        """Compute dynamic edges using percentiles P2,P20,P40,P60,P80,P98 and clamp to valid domain."""
        # Percentiles
        percentiles = [2, 20, 40, 60, 80, 98]
        reducer = ee.Reducer.percentile(percentiles)
        stats = vi_image.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=20,
            maxPixels=1e9
        )
        # Extract percentiles in order
        values = []
        for p in percentiles:
            values.append(stats.get(f"VI_p{p}").getInfo())
        # Domain clamp
        vi_domains = {
            'NDVI': (-1.0, 1.0), 'EVI': (0.0, 1.0), 'SAVI': (0.0, 1.0),
            'GNDVI': (0.0, 1.0), 'NDRE': (-1.0, 1.0), 'LSWI': (-1.0, 1.0)
        }
        lo, hi = vi_domains.get(vi_type, (-1.0, 1.0))
        edges = [max(lo, min(hi, float(v) if v is not None else lo)) for v in values]
        # Ensure monotonicity
        edges_sorted = []
        last = None
        for v in sorted(edges):
            if last is None or v >= last:
                edges_sorted.append(v)
                last = v
        # Need 6 edges -> prepend min and/or append max if needed
        # We expect 6 items from percentiles already; if duplicates collapse, we will fallback outside
        return edges_sorted

    def _labels_from_edges(self, edges: List[float]) -> List[str]:
        """Build labels like 'a to b' for consecutive pairs of 6 edges -> 5 classes."""
        def fmt(x: float) -> str:
            # round to 1–2 decimals as per policy
            s = f"{x:.2f}"
            # trim trailing zeros
            if s.endswith('0'):
                s = s.rstrip('0').rstrip('.')
            return s
        labels = []
        for i in range(5):
            labels.append(f"{fmt(edges[i])} to {fmt(edges[i+1])}")
        return labels

    def get_vi_tiles_and_legend(self, geometry: Dict, vi_type: str, start_date: str, end_date: str, mode: str = 'static') -> Dict:
        """Return XYZ tiles URL and matching discrete legend for the given VI and window.
        - Classify VI into 5 classes using edges (static/dynamic)
        - Visualize class image with values 0..4 and 5-color palette
        - Build ee map tiles URL
        """
        if not self.gee_available:
            return {"available": False, "reason": "GEE unavailable"}

        try:
            collection, ee_geometry = self.get_sentinel2_collection(geometry, start_date, end_date)
            # If empty, report unavailable
            if collection.size().getInfo() == 0:
                return {"available": False, "reason": "insufficient data for selected AOI/time"}

            # Use median composite in window for stability
            image = collection.median()
            vi_image = self.calculate_vi(image, vi_type)

            # Derive edges/palette
            static_edges, palette = self._get_static_edges_palette(vi_type)
            if mode == 'dynamic':
                try:
                    dyn_edges = self._compute_dynamic_edges(vi_image, ee_geometry, vi_type)
                    # Expect 6 edges; ensure unique and length
                    edges = dyn_edges
                    # If edges are invalid or too few/duplicated, fallback
                    if len(edges) < 6 or len(set(edges)) < 6:
                        edges = static_edges
                    else:
                        # Ensure exactly 6 by selecting first and last 6 if more
                        if len(edges) > 6:
                            edges = edges[:6]
                except Exception:
                    edges = static_edges
            else:
                edges = static_edges

            # Build class image: 0..4 for 5 bins defined by 6 edges
            e = edges
            # Build mutually exclusive class indices 0..4
            vi_clipped = vi_image.clip(ee_geometry)
            class0 = vi_clipped.gte(e[0]).And(vi_clipped.lt(e[1])).rename('cls').selfMask().multiply(0)
            class1 = vi_clipped.gte(e[1]).And(vi_clipped.lt(e[2])).rename('cls').selfMask().multiply(1)
            class2 = vi_clipped.gte(e[2]).And(vi_clipped.lt(e[3])).rename('cls').selfMask().multiply(2)
            class3 = vi_clipped.gte(e[3]).And(vi_clipped.lt(e[4])).rename('cls').selfMask().multiply(3)
            class4 = vi_clipped.gte(e[4]).And(vi_clipped.lte(e[5])).rename('cls').selfMask().multiply(4)
            class_image = class0.unmask(0).add(class1.unmask(0)).add(class2.unmask(0)).add(class3.unmask(0)).add(class4.unmask(0))

            # Visualize with min=0, max=4, palette of 5 colors
            visual = class_image.visualize(min=0, max=4, palette=palette)

            # Build mapid and token
            map_dict = visual.getMapId()
            mapid = map_dict.get('mapid')
            token = map_dict.get('token')
            tiles_url = f"https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}"

            # Compute class histogram to get percentages per class inside AOI
            try:
                # Use a pixel area weighted histogram so masked areas don't dominate
                pixel_area = ee.Image.pixelArea()
                # For each class create a mask and sum its area
                def area_for(idx):
                    return pixel_area.updateMask(class_image.eq(idx)).reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=ee_geometry,
                        scale=20,
                        maxPixels=1e9
                    ).get('area')
                areas = [area_for(i) for i in range(5)]
                areas_info = [ee.Number(a).getInfo() if a is not None else 0 for a in areas]
                total_area = float(sum([a if a is not None else 0 for a in areas_info])) or 1.0
                percents = [round((float((areas_info[i] or 0)) / total_area) * 100.0, 1) for i in range(5)]
            except Exception as hist_ex:
                print(f"Histogram failed (fallback to freq): {hist_ex}")
                try:
                    hist_dict = class_image.reduceRegion(
                        reducer=ee.Reducer.frequencyHistogram(),
                        geometry=ee_geometry,
                        scale=20,
                        maxPixels=1e9
                    ).getInfo()
                    band_name = list(hist_dict.keys())[0] if hist_dict else 'cls'
                    class_hist = hist_dict.get(band_name, {}) if hist_dict else {}
                    total_px = float(sum(class_hist.get(str(i), 0) for i in range(5))) or 1.0
                    percents = [round((float(class_hist.get(str(i), 0)) / total_px) * 100.0, 1) for i in range(5)]
                except Exception as hist_ex2:
                    print(f"Histogram freq failed: {hist_ex2}")
                    percents = [0.0, 0.0, 0.0, 0.0, 0.0]

            # Build legend breaks
            labels = self._labels_from_edges(edges)
            breaks = []
            for i in range(5):
                breaks.append({
                    "from": float(e[i]),
                    "to": float(e[i+1]),
                    "color": palette[i],
                    "label": labels[i],
                    "percent": percents[i]
                })

            return {
                "available": True,
                "tiles": tiles_url,
                "legend": {
                    "vi": vi_type,
                    "mode": "discrete",
                    "breaks": breaks
                }
            }
        except Exception as ex:
            print(f"Error building tiles and legend: {ex}")
            return {"available": False, "reason": "insufficient data for selected AOI/time"}

    def generate_vi_overlay(self, geometry: Dict, vi_type: str, date: Optional[datetime] = None) -> str:
        """Generate VI overlay image as base64 data URL"""
        try:
            if date is None:
                date = datetime.now()
            
            start_date = (date - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = (date + timedelta(days=7)).strftime('%Y-%m-%d')
            
            collection, ee_geometry = self.get_sentinel2_collection(geometry, start_date, end_date)
            image = collection.sort('system:time_start', False).first()
            
            if image is None:
                # Return empty string if no image available from GEE
                print("⚠️ No image available from GEE for overlay generation")
                return ""
            
            vi_image = self.calculate_vi(image, vi_type)
            
            # Get visualization parameters
            vis_params = self.get_vis_params(vi_type)
            
            # Generate thumbnail URL
            thumbnail_url = vi_image.getThumbURL({
                'dimensions': 512,
                'region': ee_geometry,
                'format': 'png',
                **vis_params
            })
            
            # Download and save image locally
            saved_path = image_service.save_url_image(thumbnail_url, f"vi_{vi_type}_{date.strftime('%Y%m%d')}")
            
            if saved_path:
                return image_service.get_image_url(saved_path)
            else:
                return thumbnail_url  # Fallback to direct URL
            
        except Exception as e:
            print(f"Error generating VI overlay: {e}")
            # Return empty string if overlay generation fails
            return ""
    
    def get_vis_params(self, vi_type: str) -> Dict:
        """Get visualization parameters for different VI types with improved color gradients"""
        vis_params = {
            'NDVI': {
                'min': -0.1,
                'max': 0.9,
                # Five-class, high-contrast brown→orange→yellow→light-green→dark-green
                'palette': ['#7f2704', '#f16913', '#ffffb2', '#78c679', '#006837']
            },
            'EVI': {
                'min': 0.0,
                'max': 0.8,
                # Five-class, soil/brown→orange→yellow→medium-green→dark-green
                'palette': ['#8c2d04', '#ec7014', '#fec44f', '#41ab5d', '#005a32']
            },
            'SAVI': {
                'min': 0.0,
                'max': 0.8,
                # Five-class, soil-brown→tan→yellow→light-green→green
                'palette': ['#a0522d', '#d98e4a', '#ffe082', '#74c476', '#1b7837']
            },
            'GNDVI': {
                'min': 0.0,
                'max': 0.9,
                # Five-class, yellow-green→light-green→green→dark-green (avoid purple/blue hues)
                'palette': ['#e5f5e0', '#ccebc5', '#a1d99b', '#41ab5d', '#005a32']
            },
            'NDRE': {
                'min': 0.0,
                'max': 0.5,
                # Five-class, stress/red→orange→yellow→light-green→green
                'palette': ['#d7301f', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
            },
            'LSWI': {
                'min': -0.5,
                'max': 0.6,
                # Five-class, dry/brown→tan→light-blue→blue→dark-blue (wet)
                'palette': ['#8b4513', '#f0e68c', '#a6cee3', '#1f78b4', '#0b5394']
            }
        }
        return vis_params.get(vi_type, vis_params['NDVI'])
    
    def create_statistical_overlay(self, vi_type: str, mean_value: float, min_value: float, max_value: float) -> str:
        """Create a gradient overlay based on VI statistics when GEE overlay fails"""
        try:
            # Get color parameters for this VI type
            vis_params = self.get_vis_params(vi_type)
            palette = vis_params['palette']
            vis_min = vis_params['min']
            vis_max = vis_params['max']
            
            # Normalize values to 0-1 range
            norm_mean = max(0, min(1, (mean_value - vis_min) / (vis_max - vis_min)))
            norm_min = max(0, min(1, (min_value - vis_min) / (vis_max - vis_min)))
            norm_max = max(0, min(1, (max_value - vis_min) / (vis_max - vis_min)))
            
            # Select color from palette based on mean value
            palette_index = min(len(palette) - 1, int(norm_mean * (len(palette) - 1)))
            primary_color = palette[palette_index]
            
            # Create SVG with gradient representing the VI variation
            svg_content = f"""<svg xmlns='http://www.w3.org/2000/svg' width='512' height='512' viewBox='0 0 512 512'>
                <defs>
                    <radialGradient id='viGradient' cx='50%' cy='50%' r='70%'>
                        <stop offset='0%' stop-color='{primary_color}' stop-opacity='0.8'/>
                        <stop offset='50%' stop-color='{palette[min(len(palette)-1, palette_index+1)]}' stop-opacity='0.6'/>
                        <stop offset='100%' stop-color='{palette[max(0, palette_index-1)]}' stop-opacity='0.4'/>
                    </radialGradient>
                    <pattern id='variation' patternUnits='userSpaceOnUse' width='40' height='40'>
                        <circle cx='20' cy='20' r='{int(10 + norm_mean * 15)}' fill='{primary_color}' opacity='0.3'/>
                    </pattern>
                </defs>
                <rect width='100%' height='100%' fill='url(#viGradient)'/>
                <rect width='100%' height='100%' fill='url(#variation)'/>
                <text x='10' y='30' fill='white' font-size='16' font-weight='bold'>{vi_type}</text>
                <text x='10' y='50' fill='white' font-size='12'>μ: {mean_value:.3f}</text>
                <text x='10' y='70' fill='white' font-size='12'>σ: {min_value:.3f}-{max_value:.3f}</text>
            </svg>"""
            
            # Encode as data URL
            import base64
            encoded_svg = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
            return f"data:image/svg+xml;base64,{encoded_svg}"
            
        except Exception as e:
            print(f"Error creating statistical overlay: {e}")
            # Fallback to simple colored rectangle
            return f"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='512' height='512'><rect width='100%' height='100%' fill='#ff6b6b' opacity='0.7'/><text x='10' y='30' fill='white'>{vi_type}: {mean_value:.3f}</text></svg>"
    

    
    def generate_analysis_message(self, mean_value: float, vi_type: str) -> str:
        """Generate analysis message based on VI value according to the improved classification table"""
        try:
            if vi_type == 'NDVI':
                if mean_value < 0.2:
                    return "ดินเปล่า/น้ำ - ยังไม่ปลูกหรือแปลงว่าง"
                elif 0.2 <= mean_value < 0.4:
                    return "ต้นข้าวเริ่มขึ้น - ระยะแตกกอเริ่มต้น"
                elif 0.4 <= mean_value < 0.6:
                    return "ต้นข้าวเขียวปานกลาง - ใบใบเริ่มหนาแน่น"
                else:
                    return "ต้นข้าวเขียวหนาแน่นมาก - ใบสมบูรณ์เต็มที่"
            
            elif vi_type == 'EVI':
                if mean_value < 0.2:
                    return "ยังไม่งอก - ข้าวเพิ่งปลูกหรือยังไม่ฟื้นตัว"
                elif 0.2 <= mean_value < 0.4:
                    return "ต้นกล้า - ข้าวเริ่มเขียว แตกใบ"
                elif 0.4 <= mean_value < 0.6:
                    return "เจริญเติบโต - ใบเริ่มแน่น เขียวดี"
                else:
                    return "สมบูรณ์ - ข้าวเขียวจัด ระยะออกรวง"
            
            elif vi_type == 'GNDVI':
                if mean_value < 0.3:
                    return "ขาดไนโตรเจน - ใบเหลือง ขาดความเขียว"
                elif 0.3 <= mean_value < 0.6:
                    return "ปานกลาง - ข้าวโตปกติ"
                elif 0.6 <= mean_value < 0.8:
                    return "เขียวเข้ม - ข้าวสมบูรณ์ ใบเขียวดี"
                else:
                    return "เขียวมาก - ข้าวใบแน่น ความเขียวสูง"
            
            elif vi_type == 'NDRE':
                if mean_value < 0.2:
                    return "ความเขียวต่ำ - อาจขาดไนโตรเจนหรือเริ่มต้นฤดูปลูก"
                elif 0.2 <= mean_value < 0.4:
                    return "ปานกลาง - ใบเริ่มเขียว"
                elif 0.4 <= mean_value < 0.5:
                    return "เขียวดี - พืชสมบูรณ์"
                else:
                    return "เขียวมาก - ความเขียวสูงมาก"
            
            elif vi_type == 'LSWI':
                if mean_value < 0.0:
                    return "ดินแห้ง - ความชื้นต่ำ"
                elif 0.0 <= mean_value < 0.2:
                    return "ชื้นน้อย - เริ่มขาดน้ำ"
                elif 0.2 <= mean_value < 0.35:
                    return "ชื้นปกติ - ความชื้นพอเหมาะ"
                else:
                    return "ชุ่มน้ำ - ความชื้นสูง"
            
            elif vi_type == 'SAVI':
                if mean_value < 0.2:
                    return "ดินเปล่า - พืชไม่ปกคลุมดิน"
                elif 0.2 <= mean_value < 0.4:
                    return "เริ่มปกคลุม - ข้าวโตเริ่มบังดิน"
                elif 0.4 <= mean_value < 0.6:
                    return "ปานกลาง - ข้าวเขียวและปกคลุมดี"
                else:
                    return "เขียวจัด - ข้าวเขียวหนาแน่นมาก"
                    
            else:
                return f"ค่าเฉลี่ย {vi_type}: {mean_value:.3f}"
                
        except Exception as e:
            print(f"Error generating analysis message: {e}")
            return "ไม่สามารถวิเคราะห์ได้"
    
    def get_latest_images_data(self, geometry: Dict, vi_type: str, limit: int = 4) -> List[Dict]:
        """Get the latest real images with VI data from Google Earth Engine"""
        print(f"🛰️ Fetching real satellite data from GEE for {vi_type}")
        
        # Check if GEE is available
        if not self.gee_available:
            print("❌ Google Earth Engine ไม่พร้อมใช้งาน")
            raise Exception("ไม่สามารถเชื่อมต่อ Google Earth Engine ได้ กรุณาติดต่อผู้ดูแลระบบ")
        
        try:
            
            # Get images from the last 6 months for better date diversity
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            print(f"📅 Searching for images between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
            
            collection, ee_geometry = self.get_sentinel2_collection(
                geometry, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            # Apply stronger cloud masking for cleaner images
            # Use QA60 band for more reliable cloud masking
            def mask_clouds(image):
                # Use QA60 band for cloud and cirrus masking
                qa = image.select('QA60')
                
                # Ensure QA60 is treated as integer for bitwise operations
                qa = qa.toInt()
                
                cloud_bit_mask = 1 << 10  # Clouds
                cirrus_bit_mask = 1 << 11  # Cirrus
                
                # Both flags should be set to zero, indicating clear conditions
                mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                    qa.bitwiseAnd(cirrus_bit_mask).eq(0))
                
                return image.updateMask(mask).copyProperties(image, ["system:time_start"])
            
            # Apply cloud masking to collection
            try:
                collection = collection.map(mask_clouds)
            except Exception as e:
                print(f"Cloud masking failed, using original collection: {e}")
            
            # Check if any images are available after cloud masking
            total_images_before = collection.size().getInfo()
            print(f"📊 Found {total_images_before} cloud-masked images")
            
            if total_images_before == 0:
                print("❌ No cloud-free images found in the specified date range")
                return []
            
            # Sort by date (most recent first)
            collection = collection.sort('system:time_start', False)
            
            # Get image list and process to ensure unique dates
            image_list = collection.toList(collection.size())
            total_images = image_list.size().getInfo()
            
            print(f"🔍 Processing {total_images} images to find {limit} unique dates")
            
            images_data = []
            used_dates = set()  # Track dates to ensure uniqueness
            
            # Process images until we have enough unique dates
            # Ensure we get diverse dates by checking a larger range
            max_attempts = min(total_images, limit * 5)  # Check up to 5x limit to find diverse dates
            
            for i in range(max_attempts):
                if len(images_data) >= limit:
                    break
                    
                try:
                    image = ee.Image(image_list.get(i))
                    
                    # Get date
                    date_millis = image.get('system:time_start').getInfo()
                    acquisition_date = datetime.fromtimestamp(date_millis / 1000)
                    
                    # Check if we already have data for this date (same day)
                    date_key = acquisition_date.strftime('%Y-%m-%d')
                    if date_key in used_dates:
                        print(f"⚠️ Skipping duplicate date: {date_key}")
                        continue  # Skip this image, we already have data for this date
                    
                    # Also ensure minimum gap between dates (at least 5 days apart)
                    if used_dates:
                        min_gap_found = True
                        for existing_date_str in used_dates:
                            existing_date = datetime.strptime(existing_date_str, '%Y-%m-%d')
                            gap_days = abs((acquisition_date - existing_date).days)
                            if gap_days < 5:  # Minimum 5 days gap
                                print(f"⚠️ Skipping date {date_key} - too close to {existing_date_str} (gap: {gap_days} days)")
                                min_gap_found = False
                                break
                        
                        if not min_gap_found:
                            continue
                    
                    # Calculate VI
                    vi_image = self.calculate_vi(image, vi_type)
                    
                    # Get VI statistics
                    # Use 20m for indices relying on 20m bands (e.g., NDRE uses B5, LSWI uses B11)
                    analysis_scale = 20 if vi_type in ['NDRE', 'LSWI'] else 10

                    stats = vi_image.reduceRegion(
                        reducer=ee.Reducer.mean().combine(
                            reducer2=ee.Reducer.minMax(),
                            sharedInputs=True
                        ),
                        geometry=ee_geometry,
                        scale=analysis_scale,
                        maxPixels=1e9
                    ).getInfo()
                    
                    mean_value = stats.get('VI_mean', 0)
                    min_value = stats.get('VI_min', 0)
                    max_value = stats.get('VI_max', 0)
                    
                    # Only include images with valid VI data
                    # Allow zero values for LSWI which can be negative
                    if mean_value is not None and (vi_type == 'LSWI' or mean_value != 0):
                        # Generate VI overlay for this specific image with improved visualization
                        vis_params = self.get_vis_params(vi_type)
                        
                        try:
                            # Clip VI image to field geometry to show only within field bounds
                            clipped_vi = vi_image.clip(ee_geometry)
                            
                            # Add specific visualization to ensure proper color gradients
                            visualized_vi = clipped_vi.visualize(**vis_params)
                            
                            thumbnail_url = visualized_vi.getThumbURL({
                                'dimensions': 512,
                                'region': ee_geometry,
                                'format': 'png'
                            })
                            
                            # Save image locally with unique filename
                            saved_path = image_service.save_url_image(
                                thumbnail_url, 
                                f"vi_{vi_type}_{acquisition_date.strftime('%Y%m%d_%H%M%S')}"
                            )
                            
                            overlay_url = image_service.get_image_url(saved_path) if saved_path else thumbnail_url
                            
                        except Exception as overlay_error:
                            print(f"⚠️ Overlay generation failed for {date_key}: {overlay_error}")
                            # Create a simple gradient overlay based on the statistics
                            overlay_url = self.create_statistical_overlay(vi_type, mean_value, min_value, max_value)
                        
                        images_data.append({
                            'acquisition_date': acquisition_date.isoformat(),
                            'mean_value': float(mean_value),
                            'min_value': float(min_value) if min_value else float(mean_value),
                            'max_value': float(max_value) if max_value else float(mean_value),
                            'overlay_url': overlay_url,
                            'analysis_message': self.generate_analysis_message(mean_value, vi_type)
                        })
                        
                        # Mark this date as used
                        used_dates.add(date_key)
                        print(f"✅ Added real GEE image for date: {date_key}, VI: {mean_value:.3f}")
                        
                except Exception as e:
                    print(f"❌ Error processing image {i}: {e}")
                    continue
            
            print(f"📊 Successfully retrieved {len(images_data)} real satellite images from {total_images} available")
            
            if len(images_data) == 0:
                print("⚠️ No valid VI images could be processed from GEE")
                return []
            
            # Sort by date (newest first) for consistent display
            images_data.sort(key=lambda x: x['acquisition_date'], reverse=True)
            
            return images_data
            
        except Exception as e:
            print(f"❌ Critical error accessing Google Earth Engine: {e}")
            print("💡 Please check:")
            print("   1. GEE service account key file exists")
            print("   2. GEE authentication is working")
            print("   3. Internet connection is available")
            return []

    def get_timeseries_data(self, geometry: Dict, vi_type: str, 
                           start_date: datetime, end_date: datetime, analysis_type: str = None) -> List[Dict]:
        """Get time series data for VI analysis - Monthly/Yearly averages only"""
        try:
            print(f"🛰️ GEE: Getting {vi_type} timeseries from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            print(f"📊 Analysis type: {analysis_type}")
            
            collection, ee_geometry = self.get_sentinel2_collection(
                geometry, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            # Check if we have any images
            image_count = collection.size().getInfo()
            print(f"📊 Found {image_count} Sentinel-2 images in date range")
            
            if image_count == 0:
                print("⚠️ No images found in the specified date range")
                return []
            
            timeseries_data = []
            
            # Handle different analysis types
            if analysis_type == 'ten_year_avg':
                # Calculate yearly averages (1 value per year)
                timeseries_data = self._calculate_yearly_averages(collection, ee_geometry, vi_type, start_date, end_date)
            else:
                # Calculate monthly averages (default behavior)
                timeseries_data = self._calculate_monthly_averages(collection, ee_geometry, vi_type, start_date, end_date)
            
            print(f"✅ Successfully processed {len(timeseries_data)} data points")
            return timeseries_data
            
        except Exception as e:
            print(f"❌ Error getting timeseries data: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_yearly_averages(self, collection, ee_geometry, vi_type: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Calculate yearly averages for 10-year analysis"""
        yearly_data = []
        
        current_year = start_date.year
        end_year = end_date.year
        
        while current_year <= end_year:
            try:
                print(f"📅 Processing year {current_year}")
                
                # Get year start and end dates
                year_start = f"{current_year}-01-01"
                year_end = f"{current_year}-12-31"
                
                # Filter collection for this year
                yearly_collection = collection.filterDate(year_start, year_end)
                yearly_count = yearly_collection.size().getInfo()
                
                if yearly_count > 0:
                    print(f"   Found {yearly_count} images in {current_year}")
                    
                    # Calculate mean VI for all images in this year
                    yearly_vi = yearly_collection.map(lambda img: self.calculate_vi(img, vi_type))
                    yearly_mean = yearly_vi.mean()
                    
                    # Get the mean value for the geometry
                    stats = yearly_mean.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=ee_geometry,
                        scale=10,
                        maxPixels=1e9
                    ).getInfo()
                    
                    vi_value = stats.get('VI')
                    
                    if vi_value is not None and vi_value > 0:
                        yearly_data.append({
                            'date': datetime(current_year, 1, 1).isoformat(),
                            'value': float(vi_value)
                        })
                        print(f"   ✅ {current_year}: {vi_value:.3f}")
                    else:
                        print(f"   ❌ {current_year}: No valid data")
                else:
                    print(f"   ⚠️ {current_year}: No images available")
                    
            except Exception as e:
                print(f"   ❌ Error processing year {current_year}: {e}")
            
            current_year += 1
        
        return yearly_data

    def _calculate_monthly_averages(self, collection, ee_geometry, vi_type: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Calculate monthly averages for monthly/yearly analysis"""
        monthly_data = []
        
        current_date = datetime(start_date.year, start_date.month, 1)
        end_month = datetime(end_date.year, end_date.month, 1)
        
        while current_date <= end_month:
            try:
                # Get month start and end dates
                month_start = current_date.strftime('%Y-%m-01')
                if current_date.month == 12:
                    next_month = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    next_month = current_date.replace(month=current_date.month + 1)
                month_end = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
                
                print(f"📅 Processing {current_date.strftime('%B %Y')} ({month_start} to {month_end})")
                
                # Filter collection for this month
                monthly_collection = collection.filterDate(month_start, month_end)
                monthly_count = monthly_collection.size().getInfo()
                
                if monthly_count > 0:
                    print(f"   Found {monthly_count} images in {current_date.strftime('%B %Y')}")
                    
                    # Calculate mean VI for all images in this month
                    monthly_vi = monthly_collection.map(lambda img: self.calculate_vi(img, vi_type))
                    monthly_mean = monthly_vi.mean()
                    
                    # Get the mean value for the geometry
                    stats = monthly_mean.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=ee_geometry,
                        scale=10,
                        maxPixels=1e9
                    ).getInfo()
                    
                    vi_value = stats.get('VI')
                    
                    if vi_value is not None and vi_value > 0:
                        monthly_data.append({
                            'date': current_date.isoformat(),
                            'value': float(vi_value)
                        })
                        print(f"   ✅ {current_date.strftime('%B %Y')}: {vi_value:.3f}")
                    else:
                        print(f"   ❌ {current_date.strftime('%B %Y')}: No valid data")
                else:
                    print(f"   ⚠️ {current_date.strftime('%B %Y')}: No images available")
                
            except Exception as e:
                print(f"   ❌ Error processing {current_date.strftime('%B %Y')}: {e}")
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return monthly_data
    

    


# Global instance
gee_service = GEEService()
