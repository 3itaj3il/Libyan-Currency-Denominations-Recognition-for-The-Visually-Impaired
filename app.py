import io
import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageStat, ImageFilter
import tensorflow as tf
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'libyan_currency_final.h5')
AUDIO_DIR  = os.path.join(BASE_DIR, 'static', 'audio')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ تم تحميل الموديل بنجاح!")
except Exception as e:
    print(f"❌ خطأ في تحميل الموديل: {e}")

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
    """ تحويل الصورة لرمادي وتطبيق فلتر كشف الحواف لمنع الصور العشوائية """
    gray_img = img.convert('L')
    edge_img = gray_img.filter(ImageFilter.FIND_EDGES)
    stat = ImageStat.Stat(edge_img)
    return stat.mean[0]  

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """ توليد خريطة ميزات Grad-CAM الحرارية تفاعلياً """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs], 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # حساب الاشتقاقات لمعرفة حساسية الطبقة للقرار المتوقع
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # تسوية الخريطة الحرارية (ReLU والتنظيم بين 0 و 1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def get_explanation_image(raw_img, heatmap):
    """ تطبيق الفلتر الحراري، دمج الصورتين، وتحويل النتيجة لـ Base64 """
    # إعادة تحجيم الخريطة لتطابق الصورة المعالجة
    img = np.array(raw_img.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # دمج الخريطة فوق الصورة بنسبة 60% للصورة و 40% للخريطة
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    # حفظ في الذاكرة المؤقتة للسرعة وعدم ملء القرص
    pil_img = Image.fromarray(superimposed_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

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
            'audio': "error2.mp3",
            'explanation': None
        })

    processed_img = raw_img.resize((224, 224))
    img_array = np.array(processed_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx  = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx] * 100)

    explanation_img_base64 = None

    if confidence < 65.0:
        arabic_name = "عذراً، لم يتم التأكد من العملة"
        audio_file  = "error1.mp3" 
    else:
        class_name = CLASS_FOLDERS[class_idx]
        arabic_name = CLASS_NAMES.get(class_name, class_name)
        audio_file  = AUDIO_FILES.get(class_name, '')
        
        # تفعيل جزء الذكاء الاصطناعي التفسيري (XAI) بشكل آمن وديناميكي
        try:
            last_conv_layer_name = None
            target_model = model

            # البحث داخل MobileNetV2 إذا كانت موجودة كطبقة فرعية
            for layer in reversed(model.layers):
                if hasattr(layer, 'layers'):  # طبقة فرعية (sub-model)
                    target_model = layer
                    for sub_layer in reversed(layer.layers):
                        try:
                            if len(sub_layer.output.shape) == 4:
                                last_conv_layer_name = sub_layer.name
                                break
                        except:
                            continue
                    break
                try:
                    if len(layer.output.shape) == 4:
                        last_conv_layer_name = layer.name
                        break
                except:
                    continue

            if not last_conv_layer_name:
                last_conv_layer_name = 'Conv_1'  # اسم آخر conv في MobileNetV2

            print(f"ℹ️ [XAI] تم التعرف التلقائي على الطبقة: {last_conv_layer_name}")
            
            # إنتاج الصورة المفسرة
            heatmap = make_gradcam_heatmap(img_array, target_model, last_conv_layer_name, pred_index=class_idx)
            explanation_img_base64 = get_explanation_image(raw_img, heatmap)
            
        except Exception as e:
            print(f"❌ فشل توليد تفسير القرار: {e}")

    return jsonify({
        'class': arabic_name,
        'confidence': round(confidence, 1),
        'audio': audio_file,
        'explanation': explanation_img_base64
    })

@app.route('/static/audio/<filename>')
def audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

# التعديل الهام لمنصة Render لسحب الـ Port ديناميكياً
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)