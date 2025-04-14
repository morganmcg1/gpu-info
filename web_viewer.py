#!/usr/bin/env python3
"""
Web Viewer for GPU Info Summaries

This script creates a simple web interface to view the summaries generated from YouTube videos.
"""

import os
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import markdown
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SummaryHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler for serving summaries."""
    
    def __init__(self, *args, summaries_dir=None, **kwargs):
        self.summaries_dir = summaries_dir
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate HTML for the index page
            html = self.generate_index_html()
            self.wfile.write(html.encode('utf-8'))
            return
        
        # Check if the path is a summary file
        if self.path.startswith('/summary/'):
            summary_id = self.path.split('/summary/')[1]
            summary_path = os.path.join(self.summaries_dir, f"{summary_id}_summary.md")
            
            if os.path.exists(summary_path):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                # Generate HTML for the summary
                html = self.generate_summary_html(summary_path, summary_id)
                self.wfile.write(html.encode('utf-8'))
                return
        
        # Fall back to the default handler for other paths
        return super().do_GET()
    
    def generate_index_html(self):
        """Generate HTML for the index page."""
        summaries = []
        
        # Get all summary files
        for file_path in Path(self.summaries_dir).glob('*_summary.md'):
            summary_id = file_path.stem.split('_summary')[0]
            
            # Try to extract the title from the summary file
            title = f"Summary {summary_id}"
            try:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('# Summary of '):
                        url = first_line[12:]
                        title = f"Summary of {url}"
            except Exception as e:
                logger.error(f"Error reading summary file {file_path}: {e}")
            
            summaries.append({
                'id': summary_id,
                'title': title,
                'path': file_path
            })
        
        # Sort summaries by ID
        summaries.sort(key=lambda x: x['id'])
        
        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPU Info Summaries</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1 {
                    color: #333;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                }
                ul {
                    list-style-type: none;
                    padding: 0;
                }
                li {
                    margin-bottom: 10px;
                    padding: 10px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }
                a {
                    color: #0066cc;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>GPU Info Summaries</h1>
            <ul>
        """
        
        for summary in summaries:
            html += f"""
                <li>
                    <a href="/summary/{summary['id']}">{summary['title']}</a>
                </li>
            """
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
    
    def generate_summary_html(self, summary_path, summary_id):
        """Generate HTML for a summary."""
        try:
            with open(summary_path, 'r') as f:
                content = f.read()
            
            # Convert markdown to HTML
            html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
            
            # Generate HTML
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Summary {summary_id}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    h1, h2, h3, h4, h5, h6 {{
                        color: #333;
                        margin-top: 20px;
                    }}
                    h1 {{
                        border-bottom: 1px solid #ddd;
                        padding-bottom: 10px;
                    }}
                    pre {{
                        background-color: #f5f5f5;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    code {{
                        font-family: Consolas, Monaco, 'Andale Mono', monospace;
                        background-color: #f5f5f5;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }}
                    pre code {{
                        background-color: transparent;
                        padding: 0;
                    }}
                    a {{
                        color: #0066cc;
                        text-decoration: none;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    .back-link {{
                        margin-bottom: 20px;
                    }}
                </style>
            </head>
            <body>
                <div class="back-link">
                    <a href="/">← Back to Index</a>
                </div>
                {html_content}
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating summary HTML for {summary_path}: {e}")
            return f"<html><body><h1>Error</h1><p>Failed to generate summary HTML: {e}</p></body></html>"

def run_server(host, port, summaries_dir):
    """Run the HTTP server."""
    # Create a custom handler with the summaries directory
    handler = lambda *args, **kwargs: SummaryHTTPRequestHandler(*args, summaries_dir=summaries_dir, **kwargs)
    
    # Create and start the server
    server = HTTPServer((host, port), handler)
    logger.info(f"Server started at http://{host}:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
    finally:
        server.server_close()

def main():
    parser = argparse.ArgumentParser(description='Web Viewer for GPU Info Summaries')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=12000, help='Port to bind the server to')
    parser.add_argument('--summaries_dir', type=str, default='summaries', help='Directory containing summary files')
    args = parser.parse_args()
    
    # Create summaries directory if it doesn't exist
    os.makedirs(args.summaries_dir, exist_ok=True)
    
    # Run the server
    run_server(args.host, args.port, args.summaries_dir)

if __name__ == "__main__":
    main()