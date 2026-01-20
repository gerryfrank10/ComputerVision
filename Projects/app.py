from flask import Flask, request, render_template, send_file
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    file = request.files['image']
    format = request.form['format']
    image = Image.open(file)
    img_io = io.BytesIO()
    image.save(img_io, format.upper())
    img_io.seek(0)
    return send_file(img_io, mimetype=f'image/{format}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)