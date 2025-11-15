# tests/test_visios.py
"""
Comprehensive test suite for VISIOS agent.
Run with: pytest tests/test_visios.py -v
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import numpy as np
import json

from swim.agents.visios.visios_agent import VisiosAgent


@pytest.fixture
def temp_image_dir():
    """Create a temporary directory for test images."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_images(temp_image_dir):
    """Generate sample test images."""
    images = []
    
    # Create test images with different characteristics
    for i, (r, g, b) in enumerate([
        (50, 200, 80),   # Green-dominant (simulated bloom)
        (100, 110, 120), # Balanced (clear water)
        (80, 180, 90),   # Blue-green (cyanobacteria)
        (200, 80, 70),   # Red-dominant (algae/sediment)
    ]):
        img_array = np.full((100, 100, 3), [r, g, b], dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = temp_image_dir / f"test_image_{i}.jpg"
        img.save(img_path)
        images.append(img_path.name)
    
    return images


@pytest.fixture
def agent(temp_image_dir):
    """Initialize VISIOS agent with temp directory."""
    return VisiosAgent(image_dir=temp_image_dir)


class TestVisiosAgent:
    """Test suite for core VISIOS agent functionality."""
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.image_dir.exists()
        assert agent.output_dir.exists()
        assert len(agent.supported_formats) > 0
    
    def test_list_images(self, agent, sample_images):
        """Test image listing functionality."""
        images = agent.list_images()
        assert len(images) == len(sample_images)
        assert all(img in images for img in sample_images)
    
    def test_analyze_image_success(self, agent, sample_images):
        """Test successful image analysis."""
        result = agent.analyze_image(sample_images[0])
        
        assert "error" not in result
        assert "classification" in result
        assert "bloom_probability" in result
        assert "confidence" in result
        assert "indicators" in result
        assert "color_analysis" in result
        assert "metadata" in result
        
        # Check probability range
        assert 0 <= result["bloom_probability"] <= 1
        
        # Check confidence levels
        assert result["confidence"] in ["low", "medium", "high"]
        
        # Check classification
        assert result["classification"] in [
            "Clear Water", "Possible Bloom", "Likely Bloom", "Severe Bloom"
        ]
    
    def test_analyze_nonexistent_image(self, agent):
        """Test handling of nonexistent image."""
        result = agent.analyze_image("nonexistent.jpg")
        assert "error" in result
    
    def test_color_analysis(self, agent, sample_images):
        """Test color distribution analysis."""
        result = agent.analyze_image(sample_images[0])
        color_stats = result["color_analysis"]
        
        assert "green_dominance" in color_stats
        assert "blue_green_ratio" in color_stats
        assert "texture_score" in color_stats
        assert "mean_rgb" in color_stats
        
        # Check RGB values are reasonable
        rgb = color_stats["mean_rgb"]
        assert 0 <= rgb["r"] <= 255
        assert 0 <= rgb["g"] <= 255
        assert 0 <= rgb["b"] <= 255
    
    def test_bloom_indicator_detection(self, agent, sample_images):
        """Test bloom indicator detection."""
        # Analyze green-dominant image (should have high bloom score)
        result = agent.analyze_image(sample_images[0])
        
        assert isinstance(result["indicators"], list)
        # Green image should trigger some indicators
        assert len(result["indicators"]) > 0
    
    def test_classification_levels(self, agent, sample_images):
        """Test that classifications match score ranges."""
        for img in sample_images:
            result = agent.analyze_image(img)
            score = result["bloom_probability"]
            classification = result["classification"]
            
            if score < 0.25:
                assert classification == "Clear Water"
            elif score < 0.5:
                assert classification == "Possible Bloom"
            elif score < 0.75:
                assert classification == "Likely Bloom"
            else:
                assert classification == "Severe Bloom"
    
    def test_recommendations_generated(self, agent, sample_images):
        """Test that recommendations are generated."""
        result = agent.analyze_image(sample_images[0])
        
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0
    
    def test_metadata_extraction(self, agent, sample_images):
        """Test metadata extraction."""
        result = agent.analyze_image(sample_images[0])
        metadata = result["metadata"]
        
        assert "filename" in metadata
        assert "file_size_kb" in metadata
        assert metadata["filename"] == sample_images[0]
        assert metadata["file_size_kb"] > 0
    
    def test_batch_summary(self, agent, sample_images):
        """Test batch analysis summary."""
        summary = agent.summarize_batch()
        
        assert "summary" in summary
        assert "statistics" in summary
        assert "high_risk_locations" in summary
        
        stats = summary["statistics"]
        assert stats["total_images"] == len(sample_images)
        assert "average_bloom_score" in stats
        assert "max_bloom_score" in stats
        assert "min_bloom_score" in stats
    
    def test_analysis_history(self, agent, sample_images):
        """Test analysis history tracking."""
        # Perform some analyses
        for img in sample_images[:2]:
            agent.analyze_image(img)
        
        history = agent.get_analysis_history(limit=10)
        
        assert len(history) >= 2
        assert all("image" in entry for entry in history)
        assert all("timestamp" in entry for entry in history)
        assert all("classification" in entry for entry in history)
    
    def test_report_generation_json(self, agent, sample_images):
        """Test JSON report generation."""
        report = agent.generate_report(output_format="json")
        
        # Should be valid JSON
        data = json.loads(report)
        assert "summary" in data
        assert "statistics" in data
    
    def test_report_generation_markdown(self, agent, sample_images):
        """Test Markdown report generation."""
        report = agent.generate_report(output_format="markdown")
        
        assert isinstance(report, str)
        assert "# VISIOS Analysis Report" in report
        assert "## Summary Statistics" in report
    
    def test_confidence_calculation(self, agent):
        """Test confidence level calculation."""
        # Test boundary cases
        assert agent._calculate_confidence(0.1) in ["low", "medium", "high"]
        assert agent._calculate_confidence(0.5) in ["low", "medium", "high"]
        assert agent._calculate_confidence(0.9) in ["low", "medium", "high"]
    
    def test_image_dimensions_recorded(self, agent, sample_images):
        """Test that image dimensions are recorded."""
        result = agent.analyze_image(sample_images[0])
        
        assert "image_dimensions" in result
        dims = result["image_dimensions"]
        assert "width" in dims
        assert "height" in dims
        assert "channels" in dims
        assert dims["width"] > 0
        assert dims["height"] > 0
        assert dims["channels"] == 3  # RGB


class TestVisiosTools:
    """Test suite for LangGraph tool functions."""
    
    def test_list_visios_images_tool(self, agent, sample_images):
        """Test list_visios_images tool."""
        from swim.agents.visios.tools.image_tools import list_visios_images
        
        result = list_visios_images()
        assert isinstance(result, str)
        assert "Available Images" in result or "No images" in result
    
    def test_analyze_image_by_name_tool(self, agent, sample_images):
        """Test analyze_image_by_name tool."""
        from swim.agents.visios.tools.image_tools import analyze_image_by_name
        
        result = analyze_image_by_name(sample_images[0])
        assert isinstance(result, str)
        assert "Analysis for" in result
        assert "Classification:" in result
    
    def test_summarize_all_images_tool(self, agent, sample_images):
        """Test summarize_all_images tool."""
        from swim.agents.visios.tools.image_tools import summarize_all_images
        
        result = summarize_all_images()
        assert isinstance(result, str)
        assert "Summary Report" in result or "HABs" in result
    
    def test_get_image_metadata_tool(self, agent, sample_images):
        """Test get_image_metadata tool."""
        from swim.agents.visios.tools.image_tools import get_image_metadata
        
        result = get_image_metadata(sample_images[0])
        assert isinstance(result, str)
        assert "Metadata" in result
    
    def test_get_analysis_history_tool(self, agent, sample_images):
        """Test get_analysis_history tool."""
        from swim.agents.visios.tools.image_tools import get_analysis_history
        
        # Perform some analyses first
        agent.analyze_image(sample_images[0])
        
        result = get_analysis_history()
        assert isinstance(result, str)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_directory(self, temp_image_dir):
        """Test agent behavior with no images."""
        agent = VisiosAgent(image_dir=temp_image_dir)
        
        images = agent.list_images()
        assert len(images) == 0
        
        summary = agent.summarize_batch()
        assert summary["statistics"]["total_images"] == 0
    
    def test_corrupted_image(self, agent, temp_image_dir):
        """Test handling of corrupted image file."""
        # Create a fake image file
        corrupt_path = temp_image_dir / "corrupt.jpg"
        with open(corrupt_path, 'w') as f:
            f.write("This is not an image")
        
        result = agent.analyze_image("corrupt.jpg")
        assert "error" in result
    
    def test_very_small_image(self, agent, temp_image_dir):
        """Test handling of very small images."""
        # Create 1x1 pixel image
        img_array = np.array([[[100, 150, 200]]], dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = temp_image_dir / "tiny.jpg"
        img.save(img_path)
        
        result = agent.analyze_image("tiny.jpg")
        # Should still analyze without error
        assert "classification" in result or "error" in result
    
    def test_unsupported_format(self, agent):
        """Test that unsupported formats are filtered out."""
        images = agent.list_images()
        # Should not contain .txt, .pdf, etc.
        assert all(Path(img).suffix.lower() in agent.supported_formats for img in images)


class TestIntegration:
    """Integration tests for VISIOS with other agents."""
    
    def test_contextual_data_structure(self, agent, sample_images):
        """Test contextual data linking structure."""
        result = agent.analyze_image(sample_images[0], include_context=True)
        
        # Even without real GPS, structure should be present
        if result.get("metadata", {}).get("gps_coordinates"):
            assert "context" in result
    
    def test_gps_extraction_format(self, agent):
        """Test GPS data format when present."""
        # This would require actual GPS-tagged images
        # For now, test the extraction method exists
        test_path = Path(__file__).parent / "fixtures" / "gps_image.jpg"
        if test_path.exists():
            gps = agent.extract_gps_data(test_path)
            if gps:
                assert "latitude" in gps
                assert "longitude" in gps
                assert isinstance(gps["latitude"], float)
                assert isinstance(gps["longitude"], float)


# Performance tests
class TestPerformance:
    """Test performance characteristics."""
    
    def test_batch_analysis_speed(self, agent, sample_images):
        """Test that batch analysis completes in reasonable time."""
        import time
        
        start = time.time()
        summary = agent.summarize_batch()
        duration = time.time() - start
        
        # Should complete in under 5 seconds for 4 images
        assert duration < 5.0
        assert summary["statistics"]["total_images"] == len(sample_images)
    
    def test_single_analysis_speed(self, agent, sample_images):
        """Test single image analysis speed."""
        import time
        
        start = time.time()
        result = agent.analyze_image(sample_images[0])
        duration = time.time() - start
        
        # Should complete in under 2 seconds
        assert duration < 2.0
        assert "classification" in result


# Fixtures for mock data
@pytest.fixture
def mock_gps_image(temp_image_dir):
    """Create a mock image with simulated GPS metadata."""
    from PIL.ExifTags import TAGS
    from PIL import ExifTags
    
    img = Image.new('RGB', (100, 100), color='green')
    img_path = temp_image_dir / "gps_test.jpg"
    
    # Note: Actually adding GPS EXIF is complex
    # This is a placeholder for the structure
    img.save(img_path)
    return img_path.name


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])