"""
Main entry point for the Deepfake Forensics Framework
"""

import sys
import argparse
from pathlib import Path

# Add the framework to Python path
sys.path.append(str(Path(__file__).parent))

from dff_framework.interface.gradio_app import DFFGradioInterface

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Deepfake Forensics Framework")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the interface on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and launch the interface
    app = DFFGradioInterface()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
