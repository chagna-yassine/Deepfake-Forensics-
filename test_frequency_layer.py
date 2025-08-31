"""
Test script specifically for the Frequency Layer
"""

import sys
from pathlib import Path

# Add the framework to Python path
sys.path.append(str(Path(__file__).parent))

from dff_framework.layers.frequency_layer import FrequencyLayer

def test_frequency_layer():
    """Test the frequency layer directly"""
    print("🔍 Testing Frequency Layer\n")
    
    # Create frequency layer
    frequency_layer = FrequencyLayer({
        'dct_block_size': 8,
        'fft_threshold': 0.1,
        'compression_quality_range': (30, 100)
    })
    
    print("✅ Frequency layer created successfully")
    
    # Test with a sample video
    sample_videos = [
        "GenConViT/sample_prediction_data/sample_1.mp4",
        "GenConViT/sample_prediction_data/sample_2.mp4",
        "GenConViT/sample_prediction_data/sample_3.mp4"
    ]
    
    demo_video = None
    for video_path in sample_videos:
        full_path = Path(video_path).resolve()
        if full_path.exists():
            demo_video = str(full_path)
            break
    
    if demo_video:
        print(f"📹 Testing with video: {demo_video}")
        
        try:
            # Test the frequency layer
            results = frequency_layer.analyze(demo_video)
            
            print("📊 FREQUENCY LAYER RESULTS:")
            print("=" * 50)
            
            if results.get('status') == 'success':
                print(f"✅ Analysis successful!")
                print(f"Frames Analyzed: {results.get('frames_analyzed', 0)}")
                print(f"Confidence: {results.get('confidence', 0):.3f}")
                
                # DCT Analysis
                dct_analysis = results.get('dct_analysis', {})
                print(f"DCT Compression Artifacts: {'DETECTED' if dct_analysis.get('compression_artifacts_detected') else 'None'}")
                print(f"DCT Blocks Analyzed: {dct_analysis.get('dct_blocks_analyzed', 0)}")
                
                # FFT Analysis
                fft_analysis = results.get('fft_analysis', {})
                print(f"FFT Anomalies: {'DETECTED' if fft_analysis.get('frequency_anomalies_detected') else 'None'}")
                print(f"Frequency Consistency: {fft_analysis.get('frequency_consistency', 0):.3f}")
                
                # Compression Analysis
                comp_analysis = results.get('compression_analysis', {})
                print(f"Multiple Compression: {'DETECTED' if comp_analysis.get('multiple_compression_detected') else 'None'}")
                print(f"Avg Quality: {comp_analysis.get('average_compression_quality', 0):.3f}")
                
                # Frequency Anomalies
                freq_anomalies = results.get('frequency_anomalies', {})
                print(f"Manipulation Patterns: {'DETECTED' if freq_anomalies.get('manipulation_detected') else 'None'}")
                print(f"Anomaly Score: {freq_anomalies.get('average_anomaly_score', 0):.3f}")
                
                # Summary
                summary = results.get('summary', {})
                print(f"Total Anomalies: {summary.get('total_anomalies_detected', 0)}")
                print(f"Analysis Quality: {summary.get('analysis_quality', 'unknown')}")
                
            else:
                print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("📹 No demo videos found")
        print("The frequency layer is ready to use with your own videos!")

if __name__ == "__main__":
    test_frequency_layer()
