#!/usr/bin/env python3
"""
Test script for LLM integration in Deepfake Forensics Framework
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dff_framework.layers.llm_analysis_layer import LLMAnalysisLayer

def test_llm_layer():
    """Test the LLM analysis layer"""
    print("🧪 Testing LLM Analysis Layer...")
    
    # Check for HF token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("⚠️ Warning: HF_TOKEN not found in environment variables.")
        print("   To test LLM analysis, set your Hugging Face token:")
        print("   export HF_TOKEN='your_token_here'")
        print("   or")
        print("   set HF_TOKEN=your_token_here")
        return False
    
    # Initialize the LLM layer
    llm_layer = LLMAnalysisLayer()
    
    # Create a mock forensic report
    mock_report = {
        'analysis_results': {
            'AI_Layer': {
                'status': 'success',
                'prediction': 'FAKE',
                'confidence': 0.85
            },
            'Frequency_Layer': {
                'status': 'success',
                'dct_analysis': {'anomalies_detected': 3},
                'fft_analysis': {'anomalies_detected': 2}
            },
            'Physics_Layer': {
                'status': 'success',
                'shadow_analysis': {'inconsistencies_detected': 1},
                'geometry_analysis': {'issues_detected': 0}
            },
            'Contextual_Layer': {
                'status': 'success',
                'reverse_search_analysis': {
                    'similar_images_found': 0,
                    'exact_matches': 0
                }
            }
        },
        'summary': {
            'overall_confidence': 0.75,
            'risk_level': 'HIGH',
            'total_layers': 4,
            'successful_layers': 4
        }
    }
    
    # Test the analysis
    print("📊 Running LLM analysis on mock report...")
    try:
        result = llm_layer.analyze("test_video.mp4", {'forensic_report': mock_report})
        
        if result.get('status') == 'success':
            print("✅ LLM analysis completed successfully!")
            print(f"   Expert Opinion: {result.get('expert_opinion', 'N/A')[:100]}...")
            print(f"   Final Verdict: {result.get('final_verdict', 'N/A')}")
            print(f"   Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"   Confidence: {result.get('confidence_assessment', 'N/A')}")
            return True
        else:
            print(f"❌ LLM analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during LLM analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_layer()
    if success:
        print("\n🎉 LLM integration test passed!")
    else:
        print("\n💥 LLM integration test failed!")
        print("\nTo enable LLM analysis:")
        print("1. Get a Hugging Face token from: https://huggingface.co/settings/tokens")
        print("2. Set the environment variable: HF_TOKEN=your_token_here")
        print("3. Run this test again")
