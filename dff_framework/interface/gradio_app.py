"""
Gradio interface for the Deepfake Forensics Framework
"""

import gradio as gr
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from dff_framework.core.framework import DeepfakeForensicsFramework
from dff_framework.layers.ai_layer_safe import SafeAILayer
from dff_framework.layers.metadata_layer import MetadataLayer
from dff_framework.layers.frequency_layer import FrequencyLayer
from dff_framework.layers.physics_layer import PhysicsLayer

class DFFGradioInterface:
    """
    Gradio interface for the Deepfake Forensics Framework
    """
    
    def __init__(self):
        self.framework = DeepfakeForensicsFramework()
        self._setup_layers()
    
    def _setup_layers(self):
        """Setup and register analysis layers"""
        # Register AI Layer (GenConViT - Safe version)
        ai_layer = SafeAILayer({
            'net': 'genconvit',
            'ed_weight': 'genconvit_ed_inference',
            'vae_weight': 'genconvit_vae_inference',
            'fp16': False,
            'num_frames': 15
        })
        self.framework.register_layer("AI_Layer", ai_layer)
        
        # Register Metadata Layer
        metadata_layer = MetadataLayer()
        self.framework.register_layer("Metadata_Layer", metadata_layer)
        
        # Register Frequency Layer
        frequency_layer = FrequencyLayer({
            'dct_block_size': 8,
            'fft_threshold': 0.1,
            'compression_quality_range': (30, 100)
        })
        self.framework.register_layer("Frequency_Layer", frequency_layer)
        
        # Register Physics Layer
        physics_layer = PhysicsLayer({
            'shadow_threshold': 0.3,
            'reflection_threshold': 0.2,
            'geometry_threshold': 0.1,
            'continuity_threshold': 0.4,
            'min_object_area': 1000,
            'max_objects': 10
        })
        self.framework.register_layer("Physics_Layer", physics_layer)
        
        print("Analysis layers registered successfully")
    
    def analyze_video(self, video_file, num_frames: int = 15, include_metadata: bool = True) -> tuple:
        """
        Analyze video file and return results
        
        Args:
            video_file: Uploaded video file
            num_frames: Number of frames to analyze
            include_metadata: Whether to include metadata analysis
            
        Returns:
            Tuple of (results_text, results_json, recommendations)
        """
        if video_file is None:
            return "Please upload a video file", "{}", "Please upload a video file to begin analysis."
        
        try:
            # Prepare analysis options
            options = {
                'num_frames': num_frames,
                'include_metadata': include_metadata
            }
            
            # Run analysis
            results = self.framework.analyze_video(video_file.name, options)
            
            # Format results for display
            results_text = self._format_results_text(results)
            try:
                results_json = json.dumps(results, indent=2)
            except (TypeError, ValueError) as json_error:
                results_json = json.dumps({"error": f"JSON serialization failed: {str(json_error)}", "status": "error"}, indent=2)
            recommendations = self._format_recommendations(results)
            
            return results_text, results_json, recommendations
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            error_json = json.dumps({"error": error_msg, "status": "failed"}, indent=2)
            return error_msg, error_json, error_msg
    
    def _format_results_text(self, results: Dict[str, Any]) -> str:
        """Format results as human-readable text"""
        text = "=== DEEPFAKE FORENSICS ANALYSIS REPORT ===\n\n"
        
        # Chain of custody
        coc = results.get('chain_of_custody', {})
        text += f"File: {coc.get('filename', 'Unknown')}\n"
        text += f"Analysis Time: {coc.get('analysis_timestamp', 'Unknown')}\n"
        text += f"File Hash: {coc.get('file_hash', 'Unknown')[:16]}...\n\n"
        
        # Analysis results
        analysis_results = results.get('analysis_results', {})
        
        # AI Layer results
        if 'AI_Layer' in analysis_results:
            ai_result = analysis_results['AI_Layer']
            if ai_result.get('status') == 'success':
                model_name = ai_result.get('model_info', {}).get('model_name', 'GenConViT')
                text += f"🤖 AI ANALYSIS ({model_name}):\n"
                text += f"  Prediction: {ai_result.get('prediction', 'Unknown')}\n"
                text += f"  Confidence: {ai_result.get('confidence', 0):.3f}\n"
                text += f"  Fake Probability: {ai_result.get('fake_probability', 0):.3f}\n"
                text += f"  Real Probability: {ai_result.get('real_probability', 0):.3f}\n"
                text += f"  Frames Analyzed: {ai_result.get('model_info', {}).get('num_frames_analyzed', 0)}\n"
                
                # Add warning if mock mode
                if ai_result.get('warning'):
                    text += f"  ⚠️  {ai_result.get('warning')}\n"
                
                text += "\n"
            else:
                text += f"🤖 AI ANALYSIS: ERROR - {ai_result.get('error', 'Unknown error')}\n\n"
        
        # Metadata Layer results
        if 'Metadata_Layer' in analysis_results:
            meta_result = analysis_results['Metadata_Layer']
            if meta_result.get('status') == 'success':
                text += "📋 METADATA ANALYSIS:\n"
                file_info = meta_result.get('file_info', {})
                text += f"  File Size: {file_info.get('file_size', 0):,} bytes\n"
                text += f"  Created: {file_info.get('creation_time', 'Unknown')}\n"
                text += f"  Modified: {file_info.get('modification_time', 'Unknown')}\n"
                text += "  Status: Basic analysis completed\n\n"
            else:
                text += f"📋 METADATA ANALYSIS: ERROR - {meta_result.get('error', 'Unknown error')}\n\n"
        
        # Frequency Layer results
        if 'Frequency_Layer' in analysis_results:
            freq_result = analysis_results['Frequency_Layer']
            if freq_result.get('status') == 'success':
                text += "🔍 FREQUENCY ANALYSIS:\n"
                text += f"  Frames Analyzed: {freq_result.get('frames_analyzed', 0)}\n"
                text += f"  Confidence: {freq_result.get('confidence', 0):.3f}\n"
                
                # DCT Analysis
                dct_analysis = freq_result.get('dct_analysis', {})
                text += f"  DCT Compression Artifacts: {'DETECTED' if dct_analysis.get('compression_artifacts_detected') else 'None'}\n"
                text += f"  DCT Blocks Analyzed: {dct_analysis.get('dct_blocks_analyzed', 0)}\n"
                
                # FFT Analysis
                fft_analysis = freq_result.get('fft_analysis', {})
                text += f"  FFT Anomalies: {'DETECTED' if fft_analysis.get('frequency_anomalies_detected') else 'None'}\n"
                text += f"  Frequency Consistency: {fft_analysis.get('frequency_consistency', 0):.3f}\n"
                
                # Compression Analysis
                comp_analysis = freq_result.get('compression_analysis', {})
                text += f"  Multiple Compression: {'DETECTED' if comp_analysis.get('multiple_compression_detected') else 'None'}\n"
                text += f"  Avg Quality: {comp_analysis.get('average_compression_quality', 0):.3f}\n"
                
                # Frequency Anomalies
                freq_anomalies = freq_result.get('frequency_anomalies', {})
                text += f"  Manipulation Patterns: {'DETECTED' if freq_anomalies.get('manipulation_detected') else 'None'}\n"
                text += f"  Anomaly Score: {freq_anomalies.get('average_anomaly_score', 0):.3f}\n"
                
                # Summary
                summary = freq_result.get('summary', {})
                text += f"  Total Anomalies: {summary.get('total_anomalies_detected', 0)}\n"
                text += f"  Analysis Quality: {summary.get('analysis_quality', 'unknown')}\n\n"
            else:
                text += f"🔍 FREQUENCY ANALYSIS: ERROR - {freq_result.get('error', 'Unknown error')}\n\n"
        
        # Physics Layer results
        if 'Physics_Layer' in analysis_results:
            physics_result = analysis_results['Physics_Layer']
            if physics_result.get('status') == 'success':
                text += "🌍 PHYSICS ANALYSIS:\n"
                text += f"  Frames Analyzed: {physics_result.get('frames_analyzed', 0)}\n"
                text += f"  Confidence: {physics_result.get('confidence', 0):.3f}\n"
                
                # Shadow Analysis
                shadow_analysis = physics_result.get('shadow_analysis', {})
                text += f"  Shadow Inconsistencies: {'DETECTED' if shadow_analysis.get('shadow_inconsistencies') else 'None'}\n"
                text += f"  Shadow Consistency: {shadow_analysis.get('shadow_consistency_score', 0):.3f}\n"
                
                # Reflection Analysis
                reflection_analysis = physics_result.get('reflection_analysis', {})
                text += f"  Reflection Inconsistencies: {'DETECTED' if reflection_analysis.get('reflection_inconsistencies') else 'None'}\n"
                text += f"  Reflection Consistency: {reflection_analysis.get('reflection_consistency_score', 0):.3f}\n"
                
                # Geometry Analysis
                geometry_analysis = physics_result.get('geometry_analysis', {})
                text += f"  Geometry Inconsistencies: {'DETECTED' if geometry_analysis.get('geometry_inconsistencies') else 'None'}\n"
                text += f"  Vanishing Point Consistency: {geometry_analysis.get('vanishing_point_consistency', 0):.3f}\n"
                
                # Object Continuity Analysis
                continuity_analysis = physics_result.get('continuity_analysis', {})
                text += f"  Continuity Violations: {'DETECTED' if continuity_analysis.get('continuity_violations') else 'None'}\n"
                text += f"  Object Consistency: {continuity_analysis.get('object_consistency_score', 0):.3f}\n"
                
                # Summary
                summary = physics_result.get('summary', {})
                text += f"  Total Physics Issues: {summary.get('total_physics_inconsistencies', 0)}\n"
                text += f"  Analysis Quality: {summary.get('analysis_quality', 'unknown')}\n\n"
            else:
                text += f"🌍 PHYSICS ANALYSIS: ERROR - {physics_result.get('error', 'Unknown error')}\n\n"
        
        # Summary
        summary = results.get('summary', {})
        text += "📊 SUMMARY:\n"
        text += f"  Total Layers: {summary.get('total_layers', 0)}\n"
        text += f"  Successful: {summary.get('successful_layers', 0)}\n"
        text += f"  Failed: {summary.get('failed_layers', 0)}\n"
        text += f"  Overall Confidence: {summary.get('overall_confidence', 0):.3f}\n\n"
        
        return text
    
    def _format_recommendations(self, results: Dict[str, Any]) -> str:
        """Format recommendations"""
        recommendations = results.get('recommendations', [])
        
        if not recommendations:
            return "No specific recommendations at this time."
        
        text = "RECOMMENDATIONS:\n"
        for i, rec in enumerate(recommendations, 1):
            text += f"{i}. {rec}\n"
        
        return text
    
    def create_interface(self):
        """Create and return the Gradio interface"""
        
        with gr.Blocks(
            title="Deepfake Forensics Framework",
            theme=gr.themes.Soft(),
            css="""
            .main-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .results-container {
                max-height: 600px;
                overflow-y: auto;
            }
            """
        ) as interface:
            
            gr.HTML("""
            <div class="main-header">
                <h1>🔍 Deepfake Forensics Framework</h1>
                <p>A comprehensive multi-layer analysis system for deepfake detection and forensic reporting</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Upload & Configure")
                    
                    video_input = gr.File(
                        label="Upload Video File",
                        file_types=["video"],
                        type="filepath"
                    )
                    
                    num_frames = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=15,
                        step=1,
                        label="Number of Frames to Analyze"
                    )
                    
                    include_metadata = gr.Checkbox(
                        value=True,
                        label="Include Metadata Analysis"
                    )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Video",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results")
                    
                    results_text = gr.Textbox(
                        label="Analysis Report",
                        lines=20,
                        max_lines=30,
                        container=True,
                        elem_classes=["results-container"]
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("JSON Output"):
                            results_json = gr.JSON(
                                label="Raw Results (JSON)"
                            )
                        
                        with gr.Tab("Recommendations"):
                            recommendations = gr.Textbox(
                                label="Recommendations",
                                lines=10
                            )
            
            # Event handlers
            analyze_btn.click(
                fn=self.analyze_video,
                inputs=[video_input, num_frames, include_metadata],
                outputs=[results_text, results_json, recommendations]
            )
            
            # Example section
            gr.Markdown("### 📋 Framework Layers")
            gr.Markdown("""
            **Current Implementation:**
            - 🤖 **AI Layer**: GenConViT model for deepfake detection
            - 📋 **Metadata Layer**: Basic file information analysis
            - 🔍 **Frequency Layer**: DCT/FFT analysis for compression artifacts and frequency anomalies
            - 🌍 **Physics Layer**: Shadow, reflection, geometry, and object continuity analysis
            
            **Planned Layers:**
            - 🎯 **Localization**: Spatial anomaly detection and heatmaps
            - 📊 **Explainability**: Attention maps and evidence visualization
            """)
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #e0e0e0;">
                <p><strong>Deepfake Forensics Framework v1.0</strong></p>
                <p>Built with GenConViT and multi-layer forensic analysis</p>
            </div>
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": False,
            "show_error": True
        }
        
        # Update with user parameters
        launch_params.update(kwargs)
        
        print("🚀 Launching Deepfake Forensics Framework...")
        print(f"📱 Interface will be available at: http://localhost:{launch_params['server_port']}")
        
        interface.launch(**launch_params)

def main():
    """Main function to run the Gradio interface"""
    app = DFFGradioInterface()
    app.launch()

if __name__ == "__main__":
    main()
