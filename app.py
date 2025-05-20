from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# 1) โหลด InceptionResNetV2 สำหรับสร้าง embedding
def create_inception_embedding(grayscaled_rgb):
    # grayscaled_rgb: list of (H,W,3) numpy arrays scaled [0,1]
    x = np.array([
        img_to_array(
            Image.fromarray((img * 255).astype(np.uint8))
            .resize((299, 299))
        ) for img in grayscaled_rgb
    ])
    x = preprocess_input(x)
    return inception.predict(x)

# 2) เตรียมโมเดล Inception (โหลดครั้งเดียว)
inception = InceptionResNetV2(weights='imagenet', include_top=True)

# 3) โหลดโมเดล colorization ของคุณ (PReLU_Colorization_Model.h5)
color_model = load_model('PRelu_Colorization_Model.h5', compile=False)
color_model.compile(optimizer='adam', loss='mean_squared_error')

# 4) ฟังก์ชัน colorize_image: รับ PIL Image ขาวดำ -> คืน PIL Image สี

def colorize_image(pil_gray):
    # แปลงเป็นขาวดำขนาด 256x256
    gray = pil_gray.convert('L').resize((256, 256))
    gray_arr = np.array(gray)

    # เตรียม embedding input
    rgb_input = gray2rgb(gray_arr.astype(np.uint8) / 255.0)
    embed = create_inception_embedding([rgb_input])

    # เตรียม L-channel input
    lab = rgb2lab([rgb_input])
    L = lab[:, :, :, 0]
    L = L.reshape(L.shape + (1,))

    # ทำนาย a,b channels
    ab = color_model.predict([L, embed])
    ab = ab * 128  # scale back

    # รวม L+a+b แล้วแปลงเป็น RGB
    lab_out = np.zeros((256, 256, 3))
    lab_out[:, :, 0] = L[0][:, :, 0]
    lab_out[:, :, 1:] = ab[0]
    rgb_out = lab2rgb(lab_out)

    # คืน PIL Image
    out_img = Image.fromarray((rgb_out * 255).astype(np.uint8))
    return out_img

# 5) สร้าง Flask API endpoint
@app.route('/colorize', methods=['POST'])
def colorize_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        pil_gray = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    # ประมวลผล
    pil_color = colorize_image(pil_gray)

    buf = io.BytesIO()
    pil_color.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
