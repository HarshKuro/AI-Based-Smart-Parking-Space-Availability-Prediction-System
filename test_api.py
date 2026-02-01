"""
API Test Script
===============
Test FastAPI endpoints with sample requests.

Usage:
    python test_api.py
    
Note: API server must be running (python src/api/app.py)

Author: Research Team
Date: February 2026
"""

import requests
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    logger.info("\n" + "="*60)
    logger.info("Testing /health endpoint")
    logger.info("="*60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        logger.info("✓ Health check passed")
        logger.info(f"  Status: {data['status']}")
        logger.info(f"  Model loaded: {data['model_loaded']}")
        logger.info(f"  Device: {data['device']}")
        logger.info(f"  Total requests: {data['total_requests']}")
    else:
        logger.error(f"❌ Health check failed: {response.status_code}")
    
    return response.status_code == 200


def test_predict():
    """Test prediction endpoint."""
    logger.info("\n" + "="*60)
    logger.info("Testing /api/v1/predict endpoint")
    logger.info("="*60)
    
    # Find a test image
    test_image_dir = Path("dataset/images")
    if not test_image_dir.exists():
        test_image_dir = Path("data_processed/test/images")
    
    if not test_image_dir.exists():
        logger.warning("No test images found. Skipping prediction test.")
        return False
    
    image_files = list(test_image_dir.glob("*.png"))
    if not image_files:
        logger.warning("No PNG images found. Skipping prediction test.")
        return False
    
    test_image = image_files[0]
    logger.info(f"Using test image: {test_image}")
    
    with open(test_image, 'rb') as f:
        files = {'file': (test_image.name, f, 'image/png')}
        response = requests.post(f"{API_BASE_URL}/api/v1/predict", files=files)
    
    if response.status_code == 200:
        data = response.json()
        logger.info("✓ Prediction successful")
        logger.info(f"  Total detections: {data['total_detections']}")
        logger.info(f"  Inference time: {data['inference_time_ms']:.2f} ms")
        logger.info(f"  Image size: {data['image_size']}")
        
        if data['detections']:
            logger.info("  Detections:")
            for i, det in enumerate(data['detections'][:5]):  # Show first 5
                logger.info(f"    {i+1}. {det['class_name']} (conf: {det['confidence']:.2f})")
    else:
        logger.error(f"❌ Prediction failed: {response.status_code}")
        logger.error(f"  Response: {response.text}")
    
    return response.status_code == 200


def test_availability():
    """Test availability endpoint."""
    logger.info("\n" + "="*60)
    logger.info("Testing /api/v1/availability endpoint")
    logger.info("="*60)
    
    response = requests.get(f"{API_BASE_URL}/api/v1/availability")
    
    if response.status_code == 200:
        data = response.json()
        logger.info("✓ Availability check successful")
        logger.info(f"  Total slots: {data['total']}")
        logger.info(f"  Free: {data['free']}")
        logger.info(f"  Occupied: {data['occupied']}")
        logger.info(f"  Partially free: {data['partially_free']}")
        logger.info(f"  Available: {data['available']}")
        logger.info(f"  Occupancy rate: {data['occupancy_rate']:.2%}")
        logger.info(f"  Availability rate: {data['availability_rate']:.2%}")
    else:
        logger.error(f"❌ Availability check failed: {response.status_code}")
    
    return response.status_code == 200


def test_forecast():
    """Test forecast endpoint."""
    logger.info("\n" + "="*60)
    logger.info("Testing /api/v1/forecast endpoint")
    logger.info("="*60)
    
    for horizon in [5, 10, 15, 30]:
        response = requests.get(f"{API_BASE_URL}/api/v1/forecast?horizon_minutes={horizon}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Forecast for {horizon} minutes:")
            logger.info(f"  Predicted available: {data['predicted_available']}")
            logger.info(f"  Predicted occupied: {data['predicted_occupied']}")
            logger.info(f"  Predicted occupancy: {data['predicted_occupancy_rate']:.2%}")
            logger.info(f"  Confidence: {data['confidence']}")
        else:
            logger.error(f"❌ Forecast failed for {horizon} minutes: {response.status_code}")
            return False
    
    return True


def test_stats():
    """Test statistics endpoint."""
    logger.info("\n" + "="*60)
    logger.info("Testing /api/v1/stats endpoint")
    logger.info("="*60)
    
    response = requests.get(f"{API_BASE_URL}/api/v1/stats")
    
    if response.status_code == 200:
        data = response.json()
        logger.info("✓ Statistics retrieved")
        logger.info(f"  Frames processed: {data['historical_stats']['total_frames_processed']}")
        logger.info(f"  Average occupancy: {data['historical_stats']['average_occupancy_rate']:.2%}")
        logger.info(f"  Peak occupancy: {data['historical_stats']['peak_occupancy_rate']:.2%}")
        logger.info(f"  API requests: {data['api_stats']['total_requests']}")
    else:
        logger.error(f"❌ Statistics retrieval failed: {response.status_code}")
    
    return response.status_code == 200


def main():
    """Run all API tests."""
    logger.info("="*80)
    logger.info(" SMART PARKING API TEST SUITE")
    logger.info("="*80)
    logger.info(f"API Base URL: {API_BASE_URL}")
    
    # Check if server is running
    try:
        requests.get(API_BASE_URL, timeout=2)
    except requests.exceptions.ConnectionError:
        logger.error("\n❌ API server is not running!")
        logger.error("Please start the server first: python src/api/app.py")
        return
    
    # Run tests
    results = {
        'health': test_health(),
        'predict': test_predict(),
        'availability': test_availability(),
        'forecast': test_forecast(),
        'stats': test_stats()
    }
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info(" TEST SUMMARY")
    logger.info("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<20} {status}")
    
    logger.info("-"*80)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
