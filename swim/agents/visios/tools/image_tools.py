# swim/agents/visios/tools/image_tools.py
# CORRECTED VERSION WITH ALL IMPORTS AND ERROR HANDLING

from langchain_core.tools import tool
from typing import Optional
import json
from datetime import datetime
from pathlib import Path

# Initialize agent - will be created on first use
_agent_instance = None

def get_agent():
    """Lazy initialization of agent to avoid import issues."""
    global _agent_instance
    if _agent_instance is None:
        from swim.agents.visios.visios_agent import VisiosAgent
        _agent_instance = VisiosAgent()
    return _agent_instance


@tool
def list_visios_images(query: str = "") -> str:
    """
    List all uploaded lake images available for HABs analysis.
    Returns a formatted list of image filenames.
    """
    try:
        agent = get_agent()
        images = agent.list_images()
        
        if not images:
            return "❌ No images found in visios_images folder. Please upload lake photos for analysis."
        
        return (
            f"🖼️ Available Images ({len(images)} total):\n" + 
            "\n".join(f"  {i+1}. {img}" for i, img in enumerate(images))
        )
    except Exception as e:
        return f"❌ Error listing images: {str(e)}"


@tool
def analyze_image_by_name(image_name: str) -> str:
    """
    Analyze a specific lake image for Harmful Algal Blooms (HABs).
    Provides detailed classification, bloom probability, color analysis, and recommendations.
    
    Args:
        image_name: The exact filename of the image to analyze
    
    Returns:
        Detailed analysis including classification, confidence, indicators, and recommendations
    """
    try:
        agent = get_agent()
        result = agent.analyze_image(image_name, include_context=True)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        # Format output
        output = f"""
🔍 **Analysis for {result['image']}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Classification:** {result['classification']}
**Bloom Probability:** {result['bloom_probability']:.1%}
**Confidence Level:** {result['confidence'].upper()}
**Model:** {result.get('model_version', 'VisionTransformer-EfficientNet')}

**Visual Indicators Detected:**
"""
        
        for indicator in result['indicators']:
            output += f"  • {indicator}\n"
        
        if not result['indicators']:
            output += "  • No significant bloom indicators detected\n"
        
        output += f"""
**Color Analysis:**
  • Green Dominance: {result['color_analysis']['green_dominance']:.2f}
  • Blue-Green Ratio: {result['color_analysis']['blue_green_ratio']:.2f}
  • Surface Texture: {result['color_analysis']['texture_score']:.2f}
"""
        
        # HSV features if available
        if 'hsv_features' in result['color_analysis']:
            hsv = result['color_analysis']['hsv_features']
            output += f"  • HSV Hue Mean: {hsv.get('hue_mean', 0):.1f}\n"
        
        output += f"""
**Image Details:**
  • Dimensions: {result['image_dimensions']['width']}x{result['image_dimensions']['height']}
  • File Size: {result['metadata']['file_size_kb']} KB
"""
        
        if result['metadata'].get('gps_coordinates'):
            coords = result['metadata']['gps_coordinates']
            output += f"  • Location: {coords['latitude']:.6f}°, {coords['longitude']:.6f}°\n"
            output += f"  • Map: https://www.google.com/maps?q={coords['latitude']},{coords['longitude']}\n"
        
        if result['metadata'].get('captured_at'):
            output += f"  • Captured: {result['metadata']['captured_at']}\n"
        
        if result['metadata'].get('device'):
            output += f"  • Device: {result['metadata']['device']}\n"
        
        output += "\n**Recommendations:**\n"
        for rec in result['recommendations']:
            output += f"  → {rec}\n"
        
        # Context integration if available
        if result.get('context'):
            output += f"\n**🔗 Integration with SWIM Platform:**\n"
            context = result['context']
            
            if context.get('location', {}).get('german_lake'):
                output += f"  • Identified Lake: {context['location']['german_lake']}\n"
            
            calibro = context.get('calibro_integration', {})
            if calibro.get('satellite_data_available'):
                output += f"  • ✅ CALIBRO satellite data available\n"
                if calibro.get('chlorophyll_a_estimate'):
                    output += f"  • Chlorophyll-a (satellite): {calibro['chlorophyll_a_estimate']}\n"
            
            homogen = context.get('homogen_integration', {})
            if homogen.get('nearest_sensor_km'):
                output += f"  • 📍 Nearest HOMOGEN sensor: {homogen['nearest_sensor_km']} km\n"
        
        output += f"\n*Analyzed at: {result['analyzed_at']}*"
        
        return output
    
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"


@tool
def summarize_all_images(detailed: str = "false") -> str:
    """
    Run batch analysis on all images and summarize HABs detection results.
    Provides statistics, risk distribution, and high-risk locations.
    
    Args:
        detailed: Set to "true" for detailed statistics (default: "false")
    
    Returns:
        Comprehensive summary of all image analyses
    """
    try:
        agent = get_agent()
        summary = agent.summarize_batch()
        
        if 'error' in summary:
            return f"❌ {summary['error']}"
        
        output = f"""
📊 **HABs Detection Summary Report**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Overall Statistics:**
  • Total Images Analyzed: {summary['statistics']['total_images']}
  • Average Bloom Score: {summary['statistics']['average_bloom_score']:.1%}
  • Highest Risk Score: {summary['statistics']['max_bloom_score']:.1%}
  • Lowest Risk Score: {summary['statistics']['min_bloom_score']:.1%}
"""
        
        if detailed.lower() == "true":
            output += f"  • Standard Deviation: {summary['statistics']['std_deviation']:.3f}\n"
        
        output += "\n**Classification Distribution:**\n"
        for label, count in summary['summary'].items():
            percentage = (count / summary['statistics']['total_images'] * 100) if summary['statistics']['total_images'] > 0 else 0
            bar = "█" * int(percentage / 5)
            output += f"  {label:20s} | {bar} {count} ({percentage:.1f}%)\n"
        
        # High risk locations
        if summary['high_risk_locations']:
            output += f"\n⚠️  **High Risk Locations ({len(summary['high_risk_locations'])}):**\n"
            for i, loc in enumerate(summary['high_risk_locations'][:5], 1):  # Top 5
                output += f"  {i}. {loc['image']}: {loc['classification']} (Score: {loc['score']:.1%})\n"
            
            if len(summary['high_risk_locations']) > 5:
                output += f"  ... and {len(summary['high_risk_locations']) - 5} more\n"
        else:
            output += "\n✅ No high-risk locations detected\n"
        
        # Geolocation info
        if summary['geotagged_images'] > 0:
            output += f"\n📍 {summary['geotagged_images']} image(s) contain GPS coordinates for mapping\n"
            
            if summary.get('locations'):
                output += "\n**GPS-Tagged Locations:**\n"
                for loc in summary['locations'][:3]:  # Show first 3
                    coords = loc['coords']
                    output += f"  • {loc['image']}: {coords['latitude']:.4f}°, {coords['longitude']:.4f}°\n"
        
        output += f"\n*Report generated: {summary['timestamp']}*"
        
        return output
    
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"


@tool
def get_image_metadata(image_name: str) -> str:
    """
    Extract and display metadata from a specific image including GPS, device info, and capture time.
    
    Args:
        image_name: The exact filename of the image
    
    Returns:
        Formatted metadata information
    """
    try:
        agent = get_agent()
        image_path = agent.image_dir / image_name
        
        if not image_path.exists():
            return f"❌ Image '{image_name}' not found in {agent.image_dir}"
        
        metadata = agent.extract_metadata(image_path)
        
        output = f"""
📋 **Metadata for {image_name}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**File Information:**
  • Filename: {metadata['filename']}
  • Size: {metadata['file_size_kb']} KB
"""
        
        if metadata.get('captured_at'):
            output += f"  • Captured: {metadata['captured_at']}\n"
        
        if metadata.get('device'):
            output += f"  • Device: {metadata['device']}\n"
        
        if metadata.get('gps_coordinates'):
            coords = metadata['gps_coordinates']
            output += f"""
**GPS Location:**
  • Latitude: {coords['latitude']:.6f}°
  • Longitude: {coords['longitude']:.6f}°
  • Map Link: https://www.google.com/maps?q={coords['latitude']},{coords['longitude']}
  • Google Maps: https://maps.google.com/?q={coords['latitude']},{coords['longitude']}
"""
        else:
            output += "\n**GPS Location:** Not available (no EXIF GPS data)\n"
        
        return output
    
    except Exception as e:
        return f"❌ Error extracting metadata: {str(e)}"


@tool
def get_analysis_history(limit: str = "10") -> str:
    """
    Retrieve recent image analysis history showing past classifications and scores.
    
    Args:
        limit: Number of recent analyses to retrieve (default: "10")
    
    Returns:
        Formatted history of recent analyses
    """
    try:
        limit_int = int(limit)
    except ValueError:
        limit_int = 10
    
    try:
        agent = get_agent()
        history = agent.get_analysis_history(limit=limit_int)
        
        if not history:
            return "📜 No analysis history available yet. Analyze some images first!"
        
        output = f"📜 **Recent Analysis History (Last {len(history)}):**\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, entry in enumerate(reversed(history), 1):  # Most recent first
            output += f"{i}. **{entry['image']}**\n"
            output += f"   Classification: {entry['classification']}\n"
            output += f"   Score: {entry['score']:.1%}\n"
            
            # Add GPS indicator
            if entry.get('has_gps'):
                output += f"   GPS: ✅ Available\n"
            else:
                output += f"   GPS: ❌ Not available\n"
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(entry['timestamp'])
                output += f"   Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            except:
                output += f"   Time: {entry['timestamp']}\n\n"
        
        return output
    
    except Exception as e:
        return f"❌ Error retrieving history: {str(e)}"


@tool
def generate_full_report(format_type: str = "markdown") -> str:
    """
    Generate a comprehensive analysis report of all images.
    Useful for creating documentation or sharing results.
    
    Args:
        format_type: Output format - "markdown" or "json" (default: "markdown")
    
    Returns:
        Full formatted report
    """
    try:
        agent = get_agent()
        report = agent.generate_report(output_format=format_type.lower())
        
        if format_type.lower() == "json":
            return f"```json\n{report}\n```"
        else:
            return report
    
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"


@tool
def check_bloom_risk_at_location(latitude: str, longitude: str, radius_km: str = "5") -> str:
    """
    Check bloom risk near a specific GPS location by analyzing images from that area.
    
    Args:
        latitude: Latitude coordinate (e.g., "47.5")
        longitude: Longitude coordinate (e.g., "9.5")
        radius_km: Search radius in kilometers (default: "5")
    
    Returns:
        Risk assessment for the specified location
    """
    try:
        lat = float(latitude)
        lon = float(longitude)
        radius = float(radius_km)
    except ValueError:
        return "❌ Invalid coordinates. Please provide valid numbers (e.g., latitude='47.5' longitude='9.5')"
    
    try:
        agent = get_agent()
        
        # Get all analyses with GPS data
        all_images = agent.list_images()
        nearby_analyses = []
        
        print(f"🔍 Searching {len(all_images)} images for location {lat:.4f}°, {lon:.4f}° within {radius}km...")
        
        for img_name in all_images:
            result = agent.analyze_image(img_name, include_context=False)
            
            if result.get('metadata', {}).get('gps_coordinates'):
                img_coords = result['metadata']['gps_coordinates']
                
                # Haversine-inspired distance (simplified for small distances)
                lat_diff = abs(img_coords['latitude'] - lat)
                lon_diff = abs(img_coords['longitude'] - lon)
                
                # Approximate distance in km (1 degree ≈ 111km at equator)
                approx_dist_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111
                
                if approx_dist_km <= radius:
                    nearby_analyses.append({
                        'image': img_name,
                        'distance_km': round(approx_dist_km, 2),
                        'classification': result['classification'],
                        'score': result['bloom_probability'],
                        'coords': img_coords
                    })
        
        if not nearby_analyses:
            return f"""📍 **No analyzed images found within {radius}km**

Target Location: {lat:.4f}°N, {lon:.4f}°E
Search Radius: {radius}km

💡 Suggestions:
  • Try increasing the search radius
  • Upload more geotagged images from this area
  • Check if existing images have GPS metadata
"""
        
        # Calculate aggregate risk
        avg_score = sum(a['score'] for a in nearby_analyses) / len(nearby_analyses)
        max_score = max(a['score'] for a in nearby_analyses)
        min_score = min(a['score'] for a in nearby_analyses)
        
        output = f"""
📍 **Bloom Risk Assessment**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Target Location:**
  • Coordinates: {lat:.4f}°N, {lon:.4f}°E
  • Search Radius: {radius}km
  • Map: https://www.google.com/maps?q={lat},{lon}

**Found {len(nearby_analyses)} analyzed image(s) nearby:**

"""
        
        for analysis in sorted(nearby_analyses, key=lambda x: x['distance_km']):
            output += f"📸 **{analysis['image']}**\n"
            output += f"   Distance: {analysis['distance_km']}km\n"
            output += f"   Risk: {analysis['classification']} ({analysis['score']:.1%})\n"
            output += f"   GPS: {analysis['coords']['latitude']:.4f}°, {analysis['coords']['longitude']:.4f}°\n\n"
        
        # Determine overall assessment
        overall = agent._classify_score(avg_score)
        
        output += f"""**Area Risk Summary:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Average Bloom Score: {avg_score:.1%}
  • Highest Risk: {max_score:.1%}
  • Lowest Risk: {min_score:.1%}
  • Overall Assessment: **{overall}**

"""
        
        # Add recommendations based on overall risk
        if avg_score > 0.75:
            output += "⚠️ **URGENT:** High bloom risk in this area. Avoid water contact!\n"
        elif avg_score > 0.5:
            output += "⚠️ **CAUTION:** Moderate to high bloom risk. Exercise caution.\n"
        elif avg_score > 0.25:
            output += "💡 **MONITOR:** Possible bloom conditions. Continue monitoring.\n"
        else:
            output += "✅ **SAFE:** Low bloom risk based on current data.\n"
        
        return output
    
    except Exception as e:
        return f"❌ Error checking location risk: {str(e)}"


@tool
def compare_images(image1: str, image2: str) -> str:
    """
    Compare two images side-by-side for bloom analysis.
    Useful for tracking changes over time or comparing different locations.
    
    Args:
        image1: First image filename
        image2: Second image filename
    
    Returns:
        Comparative analysis of both images
    """
    try:
        agent = get_agent()
        
        result1 = agent.analyze_image(image1, include_context=False)
        result2 = agent.analyze_image(image2, include_context=False)
        
        if "error" in result1:
            return f"❌ Error with {image1}: {result1['error']}"
        if "error" in result2:
            return f"❌ Error with {image2}: {result2['error']}"
        
        output = f"""
🔄 **Image Comparison Report**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Image 1: {image1}**
  • Classification: {result1['classification']}
  • Bloom Score: {result1['bloom_probability']:.1%}
  • Confidence: {result1['confidence'].upper()}

**Image 2: {image2}**
  • Classification: {result2['classification']}
  • Bloom Score: {result2['bloom_probability']:.1%}
  • Confidence: {result2['confidence'].upper()}

**Comparison:**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Calculate difference
        score_diff = result2['bloom_probability'] - result1['bloom_probability']
        
        if abs(score_diff) < 0.05:
            output += f"  • Status: Similar risk levels (Δ {score_diff:+.1%})\n"
        elif score_diff > 0:
            output += f"  • Status: ⚠️ Increasing risk (Δ {score_diff:+.1%})\n"
        else:
            output += f"  • Status: ✅ Decreasing risk (Δ {score_diff:+.1%})\n"
        
        # Color analysis comparison
        color1 = result1['color_analysis']
        color2 = result2['color_analysis']
        
        output += f"\n**Color Analysis Comparison:**\n"
        output += f"  • Green Dominance: {color1['green_dominance']:.2f} → {color2['green_dominance']:.2f}\n"
        output += f"  • Blue-Green Ratio: {color1['blue_green_ratio']:.2f} → {color2['blue_green_ratio']:.2f}\n"
        output += f"  • Texture Score: {color1['texture_score']:.2f} → {color2['texture_score']:.2f}\n"
        
        return output
    
    except Exception as e:
        return f"❌ Error comparing images: {str(e)}"


# Export all tools for easy import
__all__ = [
    'list_visios_images',
    'analyze_image_by_name',
    'summarize_all_images',
    'get_image_metadata',
    'get_analysis_history',
    'generate_full_report',
    'check_bloom_risk_at_location',
    'compare_images'
]