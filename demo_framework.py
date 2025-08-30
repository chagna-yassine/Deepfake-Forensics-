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
    
    framework.register_layer("AI_Layer", ai_layer)
    framework.register_layer("Metadata_Layer", metadata_layer)
    
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
