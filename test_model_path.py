#!/usr/bin/env python3
"""
Test script to verify the global MODEL_PATH works correctly
"""

import sys
import os

# Add the package directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_path():
    """Test that MODEL_PATH is accessible and points to the correct location"""
    try:
        # Import the MODEL_PATH from the package
        from ColorCorrectionPipeline import MODEL_PATH
        
        print("✅ Successfully imported MODEL_PATH from ColorCorrectionPipeline")
        print(f"📍 MODEL_PATH: {MODEL_PATH}")
        
        # Check if the file exists
        if os.path.exists(MODEL_PATH):
            print("✅ YOLO model file exists at MODEL_PATH location")
            print(f"📊 File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        else:
            print("❌ YOLO model file NOT found at MODEL_PATH location")
            return False
        
        # Test FFC initialization without providing model_path
        print("\n🧪 Testing FFC initialization without model_path...")
        from ColorCorrectionPipeline.FFC.FF_correction import FlatFieldCorrection
        
        # Create a dummy image
        import cv2
        import numpy as np
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Initialize FFC without model_path - should use global MODEL_PATH
        ffc = FlatFieldCorrection(img=dummy_img, manual_crop=False, show=False)
        
        print(f"✅ FFC initialized successfully")
        print(f"📍 FFC model_path: {ffc.model_path}")
        print(f"🤖 Manual crop: {ffc.manual_crop}")
        print(f"⚡ Model loaded: {ffc.model is not None}")
        
        if ffc.model_path == MODEL_PATH:
            print("✅ FFC is using the global MODEL_PATH as default")
        else:
            print("❌ FFC is NOT using the global MODEL_PATH")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MODEL_PATH: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_path_import_directly():
    """Test importing MODEL_PATH directly in modules"""
    try:
        print("\n🧪 Testing direct MODEL_PATH import in modules...")
        
        # Test that we can import MODEL_PATH in any module
        exec_code = """
from ColorCorrectionPipeline import MODEL_PATH
import os
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"Exists: {os.path.exists(MODEL_PATH)}")
"""
        exec(exec_code)
        print("✅ Direct MODEL_PATH import works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error with direct MODEL_PATH import: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Global MODEL_PATH Implementation")
    print("=" * 50)
    
    test1_passed = test_model_path()
    test2_passed = test_model_path_import_directly()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 All MODEL_PATH tests passed!")
        print("📦 The package correctly provides a global MODEL_PATH constant")
        sys.exit(0)
    else:
        print("💥 Some MODEL_PATH tests failed!")
        sys.exit(1)