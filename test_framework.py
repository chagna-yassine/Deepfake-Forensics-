"""
Test script to verify the Deepfake Forensics Framework setup
"""

import sys
from pathlib import Path

# Add the framework to Python path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test basic framework imports"""
    try:
        from dff_framework.core.framework import DeepfakeForensicsFramework
        from dff_framework.core.base_layer import BaseForensicLayer
        from dff_framework.layers.metadata_layer import MetadataLayer
        print("✅ Basic framework imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic framework imports failed: {e}")
        return False

def test_framework_creation():
    """Test framework creation"""
    try:
        from dff_framework.core.framework import DeepfakeForensicsFramework
        from dff_framework.layers.metadata_layer import MetadataLayer
        
        framework = DeepfakeForensicsFramework()
        metadata_layer = MetadataLayer()
        framework.register_layer("Metadata_Layer", metadata_layer)
        
        print("✅ Framework creation successful")
        return True
    except Exception as e:
        print(f"❌ Framework creation failed: {e}")
        return False

def test_gradio_import():
    """Test Gradio import"""
    try:
        import gradio as gr
        print("✅ Gradio import successful")
        return True
    except Exception as e:
        print(f"❌ Gradio import failed: {e}")
        return False

def test_genconvit_path():
    """Test GenConViT path availability"""
    try:
        genconvit_path = Path(__file__).parent / "GenConViT"
        if genconvit_path.exists():
            print(f"✅ GenConViT directory found at: {genconvit_path}")
            return True
        else:
            print(f"❌ GenConViT directory not found at: {genconvit_path}")
            return False
    except Exception as e:
        print(f"❌ GenConViT path test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Deepfake Forensics Framework Setup\n")
    
    tests = [
        ("Basic Framework Imports", test_basic_imports),
        ("Framework Creation", test_framework_creation),
        ("Gradio Import", test_gradio_import),
        ("GenConViT Path", test_genconvit_path),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing: {test_name}")
        result = test_func()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready to use.")
        print("\nTo run the interface:")
        print("python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
        if not results[2]:  # Gradio import failed
            print("\nTo install Gradio:")
            print("pip install gradio")
        
        if not results[3]:  # GenConViT path failed
            print("\nMake sure the GenConViT directory exists in the project root.")

if __name__ == "__main__":
    main()
