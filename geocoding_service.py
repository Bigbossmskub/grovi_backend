import requests
from typing import Optional
import asyncio
import aiohttp

class GeocodingService:
    def __init__(self):
        """Initialize Geocoding Service"""
        # Using Nominatim OpenStreetMap service (free)
        self.base_url = "https://nominatim.openstreetmap.org/reverse"
        
    async def reverse_geocode(self, lat: float, lng: float) -> Optional[str]:
        """Get address from latitude/longitude coordinates"""
        try:
            # Parameters for Thai language and detailed address
            params = {
                'lat': lat,
                'lon': lng,
                'format': 'json',
                'accept-language': 'th,en',  # Thai language preferred
                'zoom': 14,  # Detail level for village/tambon level
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'Rice-Field-Monitoring-System'  # Required by Nominatim
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_thai_address(data)
                    else:
                        print(f"Geocoding failed with status: {response.status}")
                        return None
                        
        except Exception as e:
            print(f"Error in reverse geocoding: {e}")
            return None
    
    def _format_thai_address(self, geocoding_data: dict) -> str:
        """Format geocoding data to Thai address format"""
        try:
            address = geocoding_data.get('address', {})
            display_name = geocoding_data.get('display_name', '')
            
            # Extract Thai address components
            components = []
            
            # Village/Moo (หมู่)
            village = address.get('village', address.get('hamlet', ''))
            if village:
                components.append(f"หมู่บ้าน{village}")
            
            # Tambon (ตำบล)  
            tambon = address.get('suburb', address.get('neighbourhood', address.get('quarter', '')))
            if tambon:
                components.append(f"ตำบล{tambon}")
            
            # Amphoe (อำเภอ)
            amphoe = address.get('city', address.get('town', address.get('municipality', '')))
            if amphoe:
                components.append(f"อำเภอ{amphoe}")
                
            # Province (จังหวัด)
            province = address.get('state', address.get('province', ''))
            if province:
                components.append(f"จังหวัด{province}")
            
            # If we have structured components, use them
            if len(components) >= 2:
                return ' '.join(components)
            
            # Fallback to parsing display_name for Thai addresses
            if 'ตำบล' in display_name or 'อำเภอ' in display_name:
                # Try to extract meaningful parts from display name
                parts = display_name.split(',')
                thai_parts = []
                
                for part in parts:
                    part = part.strip()
                    if any(thai_word in part for thai_word in ['ตำบล', 'อำเภอ', 'จังหวัด', 'หมู่บ้าน']):
                        thai_parts.append(part)
                        if len(thai_parts) >= 3:  # Limit to avoid too long address
                            break
                
                if thai_parts:
                    return ' '.join(thai_parts[:3])
            
            # Final fallback - use first few parts of display name
            parts = display_name.split(',')[:3]
            return ', '.join(part.strip() for part in parts)
            
        except Exception as e:
            print(f"Error formatting address: {e}")
            return geocoding_data.get('display_name', 'ไม่สามารถระบุตำแหน่งได้')

    def reverse_geocode_sync(self, lat: float, lng: float) -> Optional[str]:
        """Synchronous version of reverse_geocode"""
        try:
            params = {
                'lat': lat,
                'lon': lng,
                'format': 'json',
                'accept-language': 'th,en',
                'zoom': 14,
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'Rice-Field-Monitoring-System'
            }
            
            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._format_thai_address(data)
            else:
                print(f"Geocoding failed with status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error in reverse geocoding: {e}")
            return None

# Global instance
geocoding_service = GeocodingService()
