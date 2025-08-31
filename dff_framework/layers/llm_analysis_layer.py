"""
LLM Analysis Layer for Deepfake Forensics Framework
Uses Hugging Face Inference API to provide expert analysis
"""

import os
import json
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient
from .base_layer import BaseForensicLayer


class LLMAnalysisLayer(BaseForensicLayer):
    """
    LLM Analysis Layer that provides expert deepfake forensics analysis
    using Hugging Face Inference API
    """
    
    def __init__(self, name: str = "LLM_Analysis_Layer"):
        super().__init__(name)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Hugging Face Inference Client"""
        try:
            # Check for HF token
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                print("⚠️ Warning: HF_TOKEN not found. LLM analysis will be disabled.")
                print("   Set HF_TOKEN environment variable to enable LLM analysis.")
                return
            
            self.client = InferenceClient(api_key=hf_token)
            print("✅ Hugging Face Inference Client initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing HF Inference Client: {e}")
            self.client = None
    
    def analyze(self, video_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the forensic report using LLM
        
        Args:
            video_path: Path to the video file
            options: Analysis options including the forensic report
            
        Returns:
            Dictionary containing LLM analysis results
        """
        if not self.client:
            return {
                'status': 'error',
                'error': 'LLM client not initialized. Please set HF_TOKEN environment variable.',
                'llm_analysis': None,
                'expert_opinion': None,
                'confidence_assessment': None,
                'recommendations': None
            }
        
        try:
            # Get the forensic report from options
            forensic_report = options.get('forensic_report', {}) if options else {}
            
            if not forensic_report:
                return {
                    'status': 'error',
                    'error': 'No forensic report provided for LLM analysis',
                    'llm_analysis': None,
                    'expert_opinion': None,
                    'confidence_assessment': None,
                    'recommendations': None
                }
            
            # Create the system prompt
            system_prompt = self._create_system_prompt()
            
            # Create the user prompt with the forensic report
            user_prompt = self._create_user_prompt(forensic_report)
            
            # Call the LLM
            print("LLM Analysis Layer: Starting expert analysis...")
            response = self._call_llm(system_prompt, user_prompt)
            print("LLM Analysis Layer: Expert analysis completed")
            
            # Parse the response
            analysis_result = self._parse_llm_response(response)
            
            return {
                'status': 'success',
                'llm_analysis': analysis_result,
                'expert_opinion': analysis_result.get('expert_opinion', ''),
                'confidence_assessment': analysis_result.get('confidence_assessment', ''),
                'recommendations': analysis_result.get('recommendations', ''),
                'risk_level': analysis_result.get('risk_level', 'UNKNOWN'),
                'final_verdict': analysis_result.get('final_verdict', 'INCONCLUSIVE'),
                'analysis_metadata': {
                    'model_used': 'deepseek-ai/DeepSeek-V3-0324',
                    'analysis_timestamp': self._get_timestamp(),
                    'report_analyzed': True
                }
            }
            
        except Exception as e:
            print(f"LLM Analysis Layer: Error during analysis - {e}")
            return {
                'status': 'error',
                'error': f'LLM analysis failed: {str(e)}',
                'llm_analysis': None,
                'expert_opinion': None,
                'confidence_assessment': None,
                'recommendations': None
            }
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the deepfake forensics expert"""
        return """You are a world-renowned Digital Forensics Expert specializing in Deepfake Detection and Media Authentication. You have over 15 years of experience in:

- Computer Vision and Machine Learning for Media Forensics
- Digital Image and Video Analysis
- Deepfake Detection Methodologies
- Multimedia Security and Authentication
- Court Testimony in Digital Forensics Cases

Your expertise includes:
- Analyzing AI-generated content using multiple detection techniques
- Understanding compression artifacts and frequency domain analysis
- Physical consistency analysis (shadows, reflections, geometry)
- Metadata forensics and provenance analysis
- Cross-referencing with known deepfake generation methods

You are known for your meticulous, evidence-based approach and ability to explain complex technical findings in clear, actionable terms. You always provide:
1. A clear expert opinion based on the evidence
2. Confidence assessment with reasoning
3. Specific technical findings that support your conclusion
4. Practical recommendations for further investigation or action

When analyzing forensic reports, you consider:
- The reliability and consistency of different detection methods
- The strength of evidence from each analysis layer
- Potential false positives and false negatives
- The overall pattern of findings across multiple techniques
- The quality and completeness of the analysis

You respond in a professional, authoritative tone while remaining accessible to both technical and non-technical audiences."""

    def _create_user_prompt(self, forensic_report: Dict[str, Any]) -> str:
        """Create the user prompt with the forensic report"""
        
        # Extract key information from the report
        summary = forensic_report.get('summary', {})
        analysis_results = forensic_report.get('analysis_results', {})
        
        # Format the report for the LLM
        report_text = f"""
DEEPFAKE FORENSICS ANALYSIS REPORT

=== EXECUTIVE SUMMARY ===
Overall Confidence: {summary.get('overall_confidence', 'N/A')}
Risk Level: {summary.get('risk_level', 'N/A')}
Analysis Status: {summary.get('analysis_status', 'N/A')}
Total Layers Analyzed: {summary.get('total_layers', 'N/A')}
Successful Layers: {summary.get('successful_layers', 'N/A')}

=== DETAILED ANALYSIS RESULTS ===

"""
        
        # Add results from each layer
        for layer_name, layer_results in analysis_results.items():
            if isinstance(layer_results, dict):
                report_text += f"\n--- {layer_name.upper()} ---\n"
                report_text += f"Status: {layer_results.get('status', 'Unknown')}\n"
                
                # Add specific findings based on layer type
                if layer_name == 'AI_Layer':
                    prediction = layer_results.get('prediction', 'Unknown')
                    confidence = layer_results.get('confidence', 0.0)
                    report_text += f"AI Prediction: {prediction} (Confidence: {confidence:.1%})\n"
                    
                elif layer_name == 'Frequency_Layer':
                    dct_analysis = layer_results.get('dct_analysis', {})
                    fft_analysis = layer_results.get('fft_analysis', {})
                    report_text += f"DCT Anomalies: {dct_analysis.get('anomalies_detected', 0)}\n"
                    report_text += f"FFT Anomalies: {fft_analysis.get('anomalies_detected', 0)}\n"
                    
                elif layer_name == 'Physics_Layer':
                    shadow_analysis = layer_results.get('shadow_analysis', {})
                    geometry_analysis = layer_results.get('geometry_analysis', {})
                    report_text += f"Shadow Inconsistencies: {shadow_analysis.get('inconsistencies_detected', 0)}\n"
                    report_text += f"Geometry Issues: {geometry_analysis.get('issues_detected', 0)}\n"
                    
                elif layer_name == 'Contextual_Layer':
                    reverse_search = layer_results.get('reverse_search_analysis', {})
                    report_text += f"Similar Images Found: {reverse_search.get('similar_images_found', 0)}\n"
                    report_text += f"Exact Matches: {reverse_search.get('exact_matches', 0)}\n"
        
        report_text += f"""

=== ANALYSIS REQUEST ===
Please provide your expert analysis of this deepfake forensics report. Format your response using proper markdown formatting with clear headings and structure.

**Required sections:**

## 1. Expert Opinion
Your professional assessment of whether this content is authentic or manipulated

## 2. Confidence Assessment  
Your confidence level (0-100%) with detailed reasoning

## 3. Technical Findings
Key evidence that supports your conclusion

## 4. Risk Level
Overall risk assessment (LOW/MEDIUM/HIGH/CRITICAL)

## 5. Final Verdict
Clear conclusion (AUTHENTIC/MANIPULATED/INCONCLUSIVE)

## 6. Recommendations
Specific next steps or additional analysis needed

Please structure your response clearly with proper markdown formatting and provide actionable insights based on your expertise in deepfake forensics.
"""
        
        return report_text
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM using Hugging Face Inference API"""
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent, factual responses
                top_p=0.9
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            # Try to extract structured information from the response
            analysis_result = {
                'expert_opinion': response,
                'confidence_assessment': 'Not specified',
                'recommendations': 'See expert opinion',
                'risk_level': 'UNKNOWN',
                'final_verdict': 'INCONCLUSIVE'
            }
            
            # Simple keyword extraction for risk level and verdict
            response_lower = response.lower()
            
            # Extract risk level
            if 'critical' in response_lower or 'high risk' in response_lower:
                analysis_result['risk_level'] = 'CRITICAL'
            elif 'high' in response_lower and 'risk' in response_lower:
                analysis_result['risk_level'] = 'HIGH'
            elif 'medium' in response_lower or 'moderate' in response_lower:
                analysis_result['risk_level'] = 'MEDIUM'
            elif 'low' in response_lower:
                analysis_result['risk_level'] = 'LOW'
            
            # Extract final verdict
            if 'authentic' in response_lower and 'manipulated' not in response_lower:
                analysis_result['final_verdict'] = 'AUTHENTIC'
            elif 'manipulated' in response_lower or 'fake' in response_lower or 'deepfake' in response_lower:
                analysis_result['final_verdict'] = 'MANIPULATED'
            elif 'inconclusive' in response_lower or 'uncertain' in response_lower:
                analysis_result['final_verdict'] = 'INCONCLUSIVE'
            
            # Try to extract confidence percentage
            import re
            confidence_match = re.search(r'(\d+)%', response)
            if confidence_match:
                analysis_result['confidence_assessment'] = f"{confidence_match.group(1)}% confidence"
            
            return analysis_result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                'expert_opinion': response,
                'confidence_assessment': 'Parse error',
                'recommendations': 'See expert opinion',
                'risk_level': 'UNKNOWN',
                'final_verdict': 'INCONCLUSIVE'
            }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
