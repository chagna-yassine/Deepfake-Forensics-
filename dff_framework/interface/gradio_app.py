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
            return "Please upload a video file", "", ""
        
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
            results_json = json.dumps(results, indent=2)
            recommendations = self._format_recommendations(results)
            
            return results_text, results_json, recommendations
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            return error_msg, "", error_msg
    
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
            
            **Planned Layers:**
            - 🔍 **Frequency Analysis**: DCT/FFT analysis for compression artifacts
            - 🌍 **Physics Analysis**: Shadow, reflection, and geometry consistency checks
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
