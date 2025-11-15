# swim/agents/calibro/enhanced_tools/lake_finder.py
import ee
import pandas as pd
from geopy.geocoders import Nominatim
from typing import Dict, List, Tuple, Optional
import numpy as np
import requests
from datetime import datetime, timedelta

class EnhancedWaterBodyProcessor:
    """Enhanced water body discovery and processing"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="calibro_agent_v2")
        self._initialize_gee()
        
    def _initialize_gee(self):
        """Initialize Google Earth Engine"""
        try:
            ee.Initialize()
            print("✅ Earth Engine initialized.")
        except Exception:
            print("🔑 Earth Engine not initialized. Authenticating...")
            ee.Authenticate()
            ee.Initialize()
            print("✅ Earth Engine authenticated and initialized.")

    def discover_water_bodies_by_location(self, location_query: str, radius_km: float = 50) -> List[Dict]:
        """Discover water bodies near a given location"""
        print(f"🔍 Discovering water bodies near: {location_query}")
        
        # Get location coordinates
        location = self.geolocator.geocode(location_query)
        if not location:
            raise ValueError(f"Could not find location: {location_query}")
        
        lat, lon = location.latitude, location.longitude
        print(f"📍 Location found: {lat:.4f}, {lon:.4f}")
        
        # Create search area
        point = ee.Geometry.Point([lon, lat])
        search_area = point.buffer(radius_km * 1000)  # Convert km to meters
        
        # Find water bodies using JRC Global Surface Water dataset
        water_bodies = self._find_jrc_water_bodies(search_area)
        
        # Add OpenStreetMap water bodies
        water_bodies.extend(self._find_osm_water_bodies(lat, lon, radius_km))
        
        return self._deduplicate_water_bodies(water_bodies)

    def _find_jrc_water_bodies(self, search_area: ee.Geometry) -> List[Dict]:
        """Find water bodies using JRC Global Surface Water dataset"""
        try:
            # JRC Global Surface Water Occurrence dataset
            jrc_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select('occurrence')
            
            # Find permanent water bodies (occurrence > 75%)
            permanent_water = jrc_water.gt(75)
            
            # Convert to vectors and get properties
            water_vectors = permanent_water.selfMask().reduceToVectors(
                geometry=search_area,
                scale=30,
                maxPixels=1e8
            )
            
            # Get water body information
            water_list = water_vectors.getInfo()
            
            bodies = []
            for feature in water_list.get('features', [])[:20]:  # Limit to 20 largest
                geom = feature['geometry']
                if geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]
                    area_ha = self._calculate_polygon_area(coords)
                    
                    if area_ha > 1:  # Only include bodies > 1 hectare
                        centroid = self._calculate_centroid(coords)
                        bodies.append({
                            'name': f"Lake_{len(bodies)+1}",
                            'source': 'JRC_GSW',
                            'geometry': geom,
                            'centroid': centroid,
                            'area_ha': area_ha,
                            'type': 'lake'
                        })
            
            return bodies
            
        except Exception as e:
            print(f"⚠️ Error finding JRC water bodies: {e}")
            return []

    def _find_osm_water_bodies(self, lat: float, lon: float, radius_km: float) -> List[Dict]:
        """Find water bodies using OpenStreetMap Overpass API"""
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            overpass_query = f"""
            [out:json][timeout:25];
            (
              way["natural"="water"](around:{radius_km*1000},{lat},{lon});
              relation["natural"="water"](around:{radius_km*1000},{lat},{lon});
            );
            out geom;
            """
            
            response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)
            data = response.json()
            
            bodies = []
            for element in data.get('elements', [])[:15]:  # Limit results
                if element.get('type') in ['way', 'relation'] and 'geometry' in element:
                    name = element.get('tags', {}).get('name', f"OSM_Water_{len(bodies)+1}")
                    
                    # Convert OSM geometry to our format
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    if len(coords) > 3:
                        area_ha = self._calculate_polygon_area(coords)
                        if area_ha > 0.5:  # Filter small bodies
                            centroid = self._calculate_centroid(coords)
                            bodies.append({
                                'name': name,
                                'source': 'OpenStreetMap',
                                'geometry': {'type': 'Polygon', 'coordinates': [coords]},
                                'centroid': centroid,
                                'area_ha': area_ha,
                                'type': 'lake'
                            })
            
            return bodies
            
        except Exception as e:
            print(f"⚠️ Error finding OSM water bodies: {e}")
            return []

    def intelligent_water_body_selection(self, discovered_bodies: List[Dict]) -> List[Dict]:
        """Intelligently select the most suitable water bodies for analysis"""
        scored_bodies = []
        
        for body in discovered_bodies:
            score = 0
            
            # Size score (larger bodies get higher scores)
            area = body['area_ha']
            if area > 100:
                score += 10
            elif area > 50:
                score += 8
            elif area > 10:
                score += 6
            elif area > 1:
                score += 4
            
            # Name score (named bodies are likely more important)
            if body['name'] and not body['name'].startswith(('Lake_', 'OSM_Water_')):
                score += 3
            
            # Data source reliability score
            if body['source'] == 'JRC_GSW':
                score += 2
            elif body['source'] == 'OpenStreetMap':
                score += 1
            
            scored_bodies.append({**body, 'selection_score': score})
        
        # Sort by score and return top candidates
        scored_bodies.sort(key=lambda x: x['selection_score'], reverse=True)
        return scored_bodies[:10]  # Return top 10 candidates

    def process_discovered_water_body(self, water_body: Dict, start_date: str, end_date: str) -> Dict:
        """Process a discovered water body through satellite analysis"""
        print(f"🌊 Processing water body: {water_body['name']}")
        
        # Convert geometry to Earth Engine geometry
        ee_geom = ee.Geometry(water_body['geometry'])
        
        # Get satellite data
        s2_collection = self._get_satellite_data(ee_geom, start_date, end_date)
        
        count = s2_collection.size().getInfo()
        if count == 0:
            return {'status': 'no_data', 'message': 'No satellite data available'}
        
        # Apply water quality algorithms
        processed_collection = s2_collection.map(self._apply_water_quality_algorithms)
        
        # Extract time series
        time_series = self._extract_time_series(processed_collection, ee_geom, water_body['name'])
        
        # Generate summary statistics
        summary = self._generate_summary_stats(time_series)
        
        return {
            'status': 'success',
            'water_body': water_body,
            'time_series': time_series,
            'summary': summary,
            'satellite_images': count
        }

    def _get_satellite_data(self, geometry: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
        """Get Sentinel-2 data for the specified geometry and date range"""
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                .map(self._mask_clouds_and_shadows)
                .map(self._add_water_indices))

    def _apply_water_quality_algorithms(self, image: ee.Image) -> ee.Image:
        """Apply water quality algorithms to satellite image"""
        # Add chlorophyll-a estimation (NDCI-based)
        ndci = image.normalizedDifference(['B5', 'B4']).rename('NDCI')
        chl_a = ndci.multiply(30.0).add(15.0).max(0).rename('Chl_a_mg_m3')
        
        # Add turbidity estimation
        red = image.select('B4').multiply(0.0001)
        turbidity = red.multiply(20.0).max(0).rename('Turbidity_FNU')
        
        # Add water mask
        water_mask = self._create_water_mask(image)
        
        return (image.addBands([ndci, chl_a, turbidity, water_mask])
                .updateMask(water_mask))

    def _create_water_mask(self, image: ee.Image) -> ee.Image:
        """Create water mask using NDWI"""
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        
        water_mask = (ndwi.gt(0.0).And(mndwi.gt(-0.1))
                     .And(image.select('B8').lt(0.15))
                     .rename('WATER_MASK'))
        
        return water_mask

    def _extract_time_series(self, collection: ee.ImageCollection, geometry: ee.Geometry, name: str) -> pd.DataFrame:
        """Extract time series data from image collection"""
        def extract_per_image(image):
            stats = image.select(['NDCI', 'Chl_a_mg_m3', 'Turbidity_FNU']).reduceRegion(
                reducer=ee.Reducer.median(),
                geometry=geometry,
                scale=20,
                maxPixels=1e9
            )
            return ee.Feature(None, stats.combine({
                'date': image.date().format('YYYY-MM-dd'),
                'lake_name': name
            }))
        
        fc = collection.map(extract_per_image)
        features = fc.getInfo().get('features', [])
        
        data = []
        for feature in features:
            props = feature['properties']
            if all(key in props and props[key] is not None for key in ['NDCI', 'Chl_a_mg_m3', 'Turbidity_FNU']):
                data.append(props)
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df

    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the time series"""
        if df.empty:
            return {'status': 'no_data'}
        
        summary = {
            'total_observations': len(df),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            }
        }
        
        if 'Chl_a_mg_m3' in df.columns:
            summary['chlorophyll_a'] = {
                'mean': float(df['Chl_a_mg_m3'].mean()),
                'std': float(df['Chl_a_mg_m3'].std()),
                'min': float(df['Chl_a_mg_m3'].min()),
                'max': float(df['Chl_a_mg_m3'].max())
            }
        
        if 'Turbidity_FNU' in df.columns:
            summary['turbidity'] = {
                'mean': float(df['Turbidity_FNU'].mean()),
                'std': float(df['Turbidity_FNU'].std()),
                'min': float(df['Turbidity_FNU'].min()),
                'max': float(df['Turbidity_FNU'].max())
            }
        
        return summary

    # Helper methods
    def _calculate_polygon_area(self, coords: List[Tuple]) -> float:
        """Calculate area of polygon in hectares"""
        if len(coords) < 3:
            return 0
        
        x = [coord[0] for coord in coords]
        y = [coord[1] for coord in coords]
        
        area_deg2 = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] 
                                 for i in range(-1, len(x)-1)))
        
        # Convert square degrees to hectares (rough approximation)
        area_ha = area_deg2 * 785000 * 0.001  # Rough conversion
        return area_ha

    def _calculate_centroid(self, coords: List[Tuple]) -> Tuple[float, float]:
        """Calculate centroid of polygon"""
        x = sum(coord[0] for coord in coords) / len(coords)
        y = sum(coord[1] for coord in coords) / len(coords)
        return (x, y)

    def _deduplicate_water_bodies(self, water_bodies: List[Dict]) -> List[Dict]:
        """Remove duplicate water bodies based on proximity"""
        if not water_bodies:
            return []
        
        unique_bodies = []
        for body in water_bodies:
            is_duplicate = False
            body_centroid = body['centroid']
            
            for existing in unique_bodies:
                existing_centroid = existing['centroid']
                distance = ((body_centroid[0] - existing_centroid[0])**2 + 
                           (body_centroid[1] - existing_centroid[1])**2)**0.5
                
                if distance < 0.005:  # ~500m in degrees
                    is_duplicate = True
                    if body['area_ha'] > existing['area_ha']:
                        unique_bodies.remove(existing)
                        break
                    else:
                        break
            
            if not is_duplicate:
                unique_bodies.append(body)
        
        return unique_bodies

    def _mask_clouds_and_shadows(self, image: ee.Image) -> ee.Image:
        """Mask clouds and shadows in Sentinel-2 image"""
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)

    def _add_water_indices(self, image: ee.Image) -> ee.Image:
        """Add water-related indices to image"""
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
        return image.addBands([ndwi, mndwi])