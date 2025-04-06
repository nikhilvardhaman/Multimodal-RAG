from flask import Flask,send_from_directory
import os

app = Flask(__name__,static_folder='static')

# Configure upload folder and file size limit (5MB)
app.config['OUTPUT_TEXT'] = 'output_text'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_TEXT'], exist_ok=True)

# @app.route('/static/extracted_images/<filename>')
# def serve_image(filename):
#     return send_from_directory('static/extracted_images', filename)

from app import routes  # Import routes to register endpoints
