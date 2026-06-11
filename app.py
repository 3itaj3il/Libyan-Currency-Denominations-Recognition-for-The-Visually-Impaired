import io
import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageStat, ImageFilter
import tensorflow as tf

app = Flask(__name__)


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'libyan_currency_final.h5')
AUDIO_DIR  = os.path.join(BASE_DIR, 'static', 'audio')


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ تم تحميل الموديل بنجاح!")
except Exception as e:
    print(f" خطأ: {e}")

CLASS_FOLDERS = [
    '10dinars_b', '10dinars_g', '1_dinar', '20dinars_n',
    '20dinars_o', '50dinars_o', '5dinars_n_N', '5dinars_n_P',
    '5dinars_o_b', '5dinars_o_r'
]

CLASS_NAMES = {
    '10dinars_b':  'ورقة من فئة عشرة دنانير',
    '10dinars_g':  'ورقة من فئة عشرة دنانير',
    '1_dinar':     'ورقة من فئة دينار واحد',
    '20dinars_n':  'ورقة من فئة عشرين دينار الجديدة',
    '20dinars_o':  'ورقة ملغاة من فئة عشرين دينار القديمة',
    '50dinars_o':  'ورقة ملغاة من فئة خمسين دينار القديمة',
    '5dinars_n_N': 'ورقة من فئة خمسة دنانير الجديدة',
    '5dinars_n_P': 'ورقة من فئة خمسة دنانير الجديدة',
    '5dinars_o_b': 'ورقة ملغاة من فئة خمسة دنانير القديمة',
    '5dinars_o_r': 'ورقة ملغاة من فئة خمسة دنانير القديمة'
}

AUDIO_FILES = {
    '10dinars_b': '10dinars_blue.mp3', '10dinars_g': '10dinars_green.mp3',
    '1_dinar': '1dinar_blue.mp3', '20dinars_n': '20dinars_new.mp3',
    '20dinars_o': '20dinars_old.mp3', '50dinars_o': '50dinars_old.mp3',
    '5dinars_n_N': '5dinars_new_nylon.mp3', '5dinars_n_P': '5dinars_new_paper.mp3',
    '5dinars_o_b': '5dinars_old_brown.mp3', '5dinars_o_r': '5dinars_old_red.mp3'
}

def calculate_edge_density(img):
    """
    تحويل الصورة لرمادي وتطبيق فلتر كشف الحواف.
    العملة الورقية مليئة بالتفاصيل (حواف كثيرة)، بينما الخلفيات تكون أنعم.
    """
    gray_img = img.convert('L')
    edge_img = gray_img.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edge_img)
    return stat.mean[0]  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file = request.files['image']
    raw_img = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    
    edge_score = calculate_edge_density(raw_img)
    
   
    if edge_score < 6.5:
        return jsonify({
            'class': "عذراً، الصورة غير واضحة أو ليست عملة ليبية",
            'confidence': 0,
            'audio': "error2.mp3"
        })

   
    processed_img = raw_img.resize((224, 224))
    img_array = np.array(processed_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx  = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx] * 100)

    
    if confidence < 65.0:  # رفعنا النسبة قليلاً لزيادة الدقة
        arabic_name = "عذراً، لم يتم التأكد من العملة"
        audio_file  = "error1.mp3" 
    else:
        class_name = CLASS_FOLDERS[class_idx]
        arabic_name = CLASS_NAMES.get(class_name, class_name)
        audio_file  = AUDIO_FILES.get(class_name, '')

    return jsonify({
        'class': arabic_name,
        'confidence': round(confidence, 1),
        'audio': audio_file
    })

@app.route('/static/audio/<filename>')
def audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)