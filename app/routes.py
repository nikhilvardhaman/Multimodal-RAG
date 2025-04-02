import os
from flask import request, jsonify, render_template, send_from_directory
from app import app
from app.utils import encode_image,describe_image,extract_images_and_text_from_pdf,answer_query

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200

    return jsonify({'error': 'Invalid file format. Only PDF allowed'}), 400

@app.route('/query', methods=['POST'])
def query_pdf():
    data = request.get_json()
    filename = data.get("filename")
    query = data.get("query")

    if not filename or not query:
        return jsonify({"error": "Filename and query are required"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    text = extract_images_and_text_from_pdf(file_path)

    text_path = os.path.join(app.config['OUTPUT_TEXT'], "combined_text.txt")
    
    # Optionally save the combined text to a file
    with open(text_path, "w") as text_file:
        text_file.write(text)

    response = answer_query(query)

    # Modify image paths to be accessible via Flask
    image_urls = [f"/static/extracted_images/{img}" for img in response["images"]]
    
    return jsonify({
        "answer": response["answer"],
        "images": image_urls
    })

# @app.route('/extracted_images/<filename>')
# def serve_extracted_images(filename):
#     return send_from_directory("extracted_images", filename)

if __name__ == "__main__":
    app.run(debug=True)
