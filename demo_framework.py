"""
Demo script for the Deepfake Forensics Framework
"""

import sys
from pathlib import Path

# Add the framework to Python path
sys.path.append(str(Path(__file__).parent))

from dff_framework.core.framework import DeepfakeForensicsFramework
from dff_framework.layers.ai_layer_safe import SafeAILayer
from dff_framework.layers.metadata_layer import MetadataLayer
from dff_framework.layers.frequency_layer import FrequencyLayer

def demo_framework():
    """Demonstrate the framework functionality"""
    print("🔍 Deepfake Forensics Framework Demo\n")
    
    # Create framework
    framework = DeepfakeForensicsFramework()
    
    # Register layers
    print("Setting up analysis layers...")
    ai_layer = SafeAILayer({
        'net': 'genconvit',
        'ed_weight': 'genconvit_ed_inference',
        'vae_weight': 'genconvit_vae_inference',
        'fp16': False,
        'num_frames': 15
    })
    metadata_layer = MetadataLayer()
    frequency_layer = FrequencyLayer({
        'dct_block_size': 8,
        'fft_threshold': 0.1,
        'compression_quality_range': (30, 100)
    })
    
    framework.register_layer("AI_Layer", ai_layer)
    framework.register_layer("Metadata_Layer", metadata_layer)
    framework.register_layer("Frequency_Layer", frequency_layer)
    
    print("✅ Analysis layers registered successfully\n")
    
    # Demo with a sample video (if available)
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
        print(f"📹 Found demo video: {demo_video}")
        print("Running analysis...\n")
        
        # Run analysis
        results = framework.analyze_video(demo_video, {
            'num_frames': 10,
            'include_metadata': True
        })
        
        # Display results
        print("📊 ANALYSIS RESULTS:")
        print("=" * 50)
        
        # Chain of custody
        coc = results.get('chain_of_custody', {})
        print(f"File: {coc.get('filename', 'Unknown')}")
        print(f"Analysis Time: {coc.get('analysis_timestamp', 'Unknown')}")
        print(f"File Hash: {coc.get('file_hash', 'Unknown')[:16]}...")
        print()
        
        # Analysis results
        analysis_results = results.get('analysis_results', {})
        
        # AI Layer results
        if 'AI_Layer' in analysis_results:
            ai_result = analysis_results['AI_Layer']
            if ai_result.get('status') == 'success':
                model_name = ai_result.get('model_info', {}).get('model_name', 'GenConViT')
                print(f"🤖 AI ANALYSIS ({model_name}):")
                print(f"  Prediction: {ai_result.get('prediction', 'Unknown')}")
                print(f"  Confidence: {ai_result.get('confidence', 0):.3f}")
                print(f"  Fake Probability: {ai_result.get('fake_probability', 0):.3f}")
                print(f"  Real Probability: {ai_result.get('real_probability', 0):.3f}")
                
                if ai_result.get('warning'):
                    print(f"  ⚠️  {ai_result.get('warning')}")
                print()
            else:
                print(f"🤖 AI ANALYSIS: ERROR - {ai_result.get('error', 'Unknown error')}")
        
        # Metadata Layer results
        if 'Metadata_Layer' in analysis_results:
            meta_result = analysis_results['Metadata_Layer']
            if meta_result.get('status') == 'success':
                print("📋 METADATA ANALYSIS:")
                file_info = meta_result.get('file_info', {})
                print(f"  File Size: {file_info.get('file_size', 0):,} bytes")
                print(f"  Created: {file_info.get('creation_time', 'Unknown')}")
                print(f"  Modified: {file_info.get('modification_time', 'Unknown')}")
                print()
            else:
                print(f"📋 METADATA ANALYSIS: ERROR - {meta_result.get('error', 'Unknown error')}")
        
        # Frequency Layer results
        if 'Frequency_Layer' in analysis_results:
            freq_result = analysis_results['Frequency_Layer']
            if freq_result.get('status') == 'success':
                print("🔍 FREQUENCY ANALYSIS:")
                print(f"  Frames Analyzed: {freq_result.get('frames_analyzed', 0)}")
                print(f"  Confidence: {freq_result.get('confidence', 0):.3f}")
                
                # DCT Analysis
                dct_analysis = freq_result.get('dct_analysis', {})
                print(f"  DCT Compression Artifacts: {'DETECTED' if dct_analysis.get('compression_artifacts_detected') else 'None'}")
                print(f"  DCT Blocks Analyzed: {dct_analysis.get('dct_blocks_analyzed', 0)}")
                
                # FFT Analysis
                fft_analysis = freq_result.get('fft_analysis', {})
                print(f"  FFT Anomalies: {'DETECTED' if fft_analysis.get('frequency_anomalies_detected') else 'None'}")
                print(f"  Frequency Consistency: {fft_analysis.get('frequency_consistency', 0):.3f}")
                
                # Compression Analysis
                comp_analysis = freq_result.get('compression_analysis', {})
                print(f"  Multiple Compression: {'DETECTED' if comp_analysis.get('multiple_compression_detected') else 'None'}")
                print(f"  Avg Quality: {comp_analysis.get('average_compression_quality', 0):.3f}")
                
                # Frequency Anomalies
                freq_anomalies = freq_result.get('frequency_anomalies', {})
                print(f"  Manipulation Patterns: {'DETECTED' if freq_anomalies.get('manipulation_detected') else 'None'}")
                print(f"  Anomaly Score: {freq_anomalies.get('average_anomaly_score', 0):.3f}")
                
                # Summary
                summary = freq_result.get('summary', {})
                print(f"  Total Anomalies: {summary.get('total_anomalies_detected', 0)}")
                print(f"  Analysis Quality: {summary.get('analysis_quality', 'unknown')}")
                print()
            else:
                print(f"🔍 FREQUENCY ANALYSIS: ERROR - {freq_result.get('error', 'Unknown error')}")
        
        # Summary
        summary = results.get('summary', {})
        print("📊 SUMMARY:")
        print(f"  Total Layers: {summary.get('total_layers', 0)}")
        print(f"  Successful: {summary.get('successful_layers', 0)}")
        print(f"  Failed: {summary.get('failed_layers', 0)}")
        print(f"  Overall Confidence: {summary.get('overall_confidence', 0):.3f}")
        print()
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
    else:
        print("📹 No demo videos found in sample_prediction_data directory")
        print("The framework is ready to use with your own videos!")
    
    print("\n🚀 To run the web interface:")
    print("python main.py")
    print("\n🌐 The interface will be available at: http://localhost:7860")

if __name__ == "__main__":
    demo_framework()
