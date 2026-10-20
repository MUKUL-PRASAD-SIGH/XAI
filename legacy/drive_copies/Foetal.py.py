import os
import io
import base64
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask


import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

import os
import io
import base64
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# >>> ADD THESE FUNCTIONS HERE <<<
import requests
import warnings

warnings.filterwarnings("ignore")


def analyze_gradcam_smart(gradcam_img, prediction, color_percentages):
    """Smart rule-based AI analysis - works without external APIs"""
    try:
        medical_insights = {
            'intracranial-tumor': {
                'high_red': 'High activation regions indicate potential tumor mass locations with significant diagnostic importance.',
                'high_yellow': 'Moderate activation suggests tumor margins or surrounding edema areas.',
                'high_green': 'Low activation in normal brain tissue, indicating preserved healthy regions.',
                'high_blue': 'Minimal activation in background areas, as expected in tumor cases.'
            },
            'normal': {
                'high_red': 'Unexpected high activation may indicate false positive regions requiring review.',
                'high_yellow': 'Moderate activation in normal anatomical structures.',
                'high_green': 'Standard brain tissue activation patterns within normal limits.',
                'high_blue': 'Background regions showing expected minimal activation.'
            },
            'intracranial-hemorrdge': {
                'high_red': 'Critical activation indicating hemorrhage location and extent.',
                'high_yellow': 'Moderate activation around bleeding areas or hematoma margins.',
                'high_green': 'Low activation in unaffected brain regions.',
                'high_blue': 'Background areas with minimal diagnostic significance.'
            },
            'mild-ventriculomegaly': {
                'high_red': 'High activation in ventricular regions indicating enlarged ventricles.',
                'high_yellow': 'Moderate activation around ventricular margins.',
                'high_green': 'Low activation in normal brain parenchyma.',
                'high_blue': 'Background regions with minimal diagnostic relevance.'
            },
            'moderate-ventriculomegaly': {
                'high_red': 'High activation in moderately enlarged ventricular system.',
                'high_yellow': 'Moderate activation around ventricular boundaries.',
                'high_green': 'Low activation in surrounding brain tissue.',
                'high_blue': 'Background areas with minimal diagnostic importance.'
            },
            'severe-ventriculomegaly': {
                'high_red': 'Critical activation indicating severely enlarged ventricular system.',
                'high_yellow': 'Moderate activation around distended ventricles.',
                'high_green': 'Low activation in compressed brain tissue.',
                'high_blue': 'Background areas showing minimal activation.'
            },
            'anold-chiari-malformation': {
                'high_red': 'High activation in cerebellar and brainstem regions indicating malformation.',
                'high_yellow': 'Moderate activation around posterior fossa structures.',
                'high_green': 'Low activation in normal cerebral regions.',
                'high_blue': 'Background areas with minimal diagnostic relevance.'
            },
            'arachnoid-cyst': {
                'high_red': 'High activation indicating cyst location and mass effect.',
                'high_yellow': 'Moderate activation around cyst margins.',
                'high_green': 'Low activation in normal brain parenchyma.',
                'high_blue': 'Background regions showing minimal activation.'
            },
            'cerebellah-hypoplasia': {
                'high_red': 'High activation in underdeveloped cerebellar regions.',
                'high_yellow': 'Moderate activation around affected cerebellar areas.',
                'high_green': 'Low activation in preserved brain structures.',
                'high_blue': 'Background areas with minimal diagnostic significance.'
            },
            'colphocephaly': {
                'high_red': 'High activation in enlarged occipital horns of lateral ventricles.',
                'high_yellow': 'Moderate activation around ventricular abnormalities.',
                'high_green': 'Low activation in normal brain regions.',
                'high_blue': 'Background areas showing minimal activation.'
            },
            'encephalocele': {
                'high_red': 'High activation indicating brain tissue herniation through skull defect.',
                'high_yellow': 'Moderate activation around herniated brain tissue.',
                'high_green': 'Low activation in normal intracranial structures.',
                'high_blue': 'Background regions with minimal diagnostic importance.'
            },
            'holoprosencephaly': {
                'high_red': 'High activation in undivided forebrain structures.',
                'high_yellow': 'Moderate activation around malformed brain regions.',
                'high_green': 'Low activation in preserved brain tissue.',
                'high_blue': 'Background areas with minimal diagnostic relevance.'
            },
            'hydracenphaly': {
                'high_red': 'High activation in fluid-filled spaces replacing cerebral hemispheres.',
                'high_yellow': 'Moderate activation around preserved brain structures.',
                'high_green': 'Low activation in remaining brain tissue.',
                'high_blue': 'Background regions showing minimal activation.'
            },
            'm-magna': {
                'high_red': 'High activation in enlarged cisterna magna region.',
                'high_yellow': 'Moderate activation around posterior fossa structures.',
                'high_green': 'Low activation in normal cerebellar regions.',
                'high_blue': 'Background areas with minimal diagnostic significance.'
            },
            'polencephaly': {
                'high_red': 'High activation indicating cystic cavities within brain parenchyma.',
                'high_yellow': 'Moderate activation around porencephalic cysts.',
                'high_green': 'Low activation in preserved brain tissue.',
                'high_blue': 'Background regions with minimal diagnostic importance.'
            },
            'vein-of-galen': {
                'high_red': 'High activation indicating vascular malformation in deep brain structures.',
                'high_yellow': 'Moderate activation around abnormal vascular connections.',
                'high_green': 'Low activation in normal brain parenchyma.',
                'high_blue': 'Background areas showing minimal activation.'
            }
        }

        # Determine dominant activation pattern
        dominant_color = max(color_percentages, key=color_percentages.get)
        activation_level = 'high' if color_percentages[dominant_color] > 25 else 'moderate'
        pattern_key = f"{activation_level}_{dominant_color}"

        # Get medical insight
        condition_insights = medical_insights.get(prediction, {})
        analysis = condition_insights.get(pattern_key, f"Standard {dominant_color} activation pattern observed.")

        # Enhanced analysis
        confidence_level = 'high' if color_percentages['red'] > 30 else 'moderate'
        detailed_analysis = f"{analysis} Primary activation in {dominant_color} regions ({color_percentages[dominant_color]:.1f}%) indicates {confidence_level} diagnostic confidence for {prediction.replace('-', ' ')}."

        return detailed_analysis

    except Exception as e:
        return f"AI analysis error: {str(e)}"


def get_color_percentages(pil_image):
    # Convert PIL Image to OpenCV format (BGR)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]

    # Define color ranges in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([50, 50, 50])
    upper_green = np.array([70, 255, 255])

    lower_yellow = np.array([18, 50, 50])
    upper_yellow = np.array([30, 255, 255])

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([120, 255, 255])

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Calculate percentages
    percent_red = (cv2.countNonZero(mask_red) / total_pixels) * 100
    percent_green = (cv2.countNonZero(mask_green) / total_pixels) * 100
    percent_yellow = (cv2.countNonZero(mask_yellow) / total_pixels) * 100
    percent_blue = (cv2.countNonZero(mask_blue) / total_pixels) * 100

    return {
        "red": percent_red,
        "green": percent_green,
        "yellow": percent_yellow,
        "blue": percent_blue
    }

class_names = [
    "anold-chiari-malformation",
    "arachnoid-cyst",
    "cerebellah-hypoplasia",
    "colphocephaly",
    "encephalocele",
    "holoprosencephaly",
    "hydracenphaly",
    "intracranial-hemorrdge",
    "intracranial-tumor",
    "m-magna",
    "mild-ventriculomegaly",
    "moderate-ventriculomegaly",
    "normal",
    "polencephaly",
    "severe-ventriculomegaly",
    "vein-of-galen"
]

# Full disease explanations dictionary
disease_explanations = {
    "anold-chiari-malformation": {
        "info": "Arnold-Chiari malformation is a structural defect in the cerebellum, affecting balance and coordination. It may cause headaches, dizziness, and neurological symptoms.",
        "curability": "Sometimes curable with surgery, depending on type and severity."
    },
    "arachnoid-cyst": {
        "info": "An arachnoid cyst is a fluid-filled sac located between the brain or spinal cord and the arachnoid membrane. Most are benign but can cause symptoms if they press on neural structures.",
        "curability": "Often treatable; surgery may be needed if symptomatic."
    },
    "cerebellah-hypoplasia": {
        "info": "Cerebellar hypoplasia is the underdevelopment or incomplete development of the cerebellum. It can lead to problems with movement, balance, and coordination.",
        "curability": "Not curable; therapy focuses on symptom management."
    },
    "colphocephaly": {
        "info": "Colpocephaly is a condition where the occipital horns of the lateral ventricles are abnormally enlarged, often associated with developmental delays and neurological issues.",
        "curability": "Not curable; supportive therapies may help with symptoms."
    },
    "encephalocele": {
        "info": "Encephalocele is a neural tube defect characterized by sac-like protrusions of the brain and membranes through openings in the skull.",
        "curability": "Sometimes surgically repairable; prognosis depends on severity and brain involvement."
    },
    "holoprosencephaly": {
        "info": "Holoprosencephaly is a birth defect where the forebrain fails to divide into two hemispheres, leading to facial and neurological abnormalities.",
        "curability": "Not curable; supportive care only."
    },
    "hydracenphaly": {
        "info": "Hydranencephaly is a rare condition where the brain's cerebral hemispheres are absent and replaced by sacs filled with cerebrospinal fluid.",
        "curability": "Not curable; supportive care only."
    },
    "intracranial-hemorrdge": {
        "info": "Intracranial hemorrhage is bleeding within the skull, which can damage brain tissue and is a medical emergency.",
        "curability": "Sometimes treatable depending on severity and timing."
    },
    "intracranial-tumor": {
        "info": "An intracranial tumor is an abnormal growth of cells within the brain, which can be benign or malignant and may cause various neurological symptoms.",
        "curability": "Sometimes curable depending on tumor type and treatment."
    },
    "m-magna": {
        "info": "Mega cisterna magna is an enlargement of the cisterna magna, a fluid-filled space at the base of the brain, usually benign but may be associated with other abnormalities.",
        "curability": "Usually benign and requires no treatment."
    },
    "mild-ventriculomegaly": {
        "info": "Mild ventriculomegaly is a slight enlargement of the brain's ventricles, which can be a normal variant or associated with developmental issues.",
        "curability": "Sometimes resolves on its own; may require monitoring."
    },
    "moderate-ventriculomegaly": {
        "info": "Moderate ventriculomegaly is a moderate enlargement of the brain's ventricles, possibly indicating underlying brain abnormalities.",
        "curability": "Sometimes treatable if hydrocephalus develops."
    },
    "normal": {
        "info": "No abnormality detected. The brain appears healthy and within normal limits.",
        "curability": "Not applicable."
    },
    "polencephaly": {
        "info": "Porencephaly is a rare disorder involving cysts or cavities within the brain, which can cause seizures and developmental delays.",
        "curability": "Not curable; supportive care and therapy."
    },
    "severe-ventriculomegaly": {
        "info": "Severe ventriculomegaly is a significant enlargement of the brain's ventricles, often associated with hydrocephalus and developmental issues.",
        "curability": "Sometimes treatable with surgery (e.g., shunt) if hydrocephalus occurs."
    },
    "vein-of-galen": {
        "info": "Vein of Galen malformation is a rare vascular defect in the brain that can lead to heart failure and neurological symptoms in newborns.",
        "curability": "Sometimes treatable with endovascular surgery; prognosis varies."
    }
}

app = Flask(__name__)


MODEL_PATHS = {
    "keras": "models/best_model31.keras",
    "keras1": "models/best_model24.keras",
    "keras2": "models/best_model2.keras"
}
models = {name: load_model(path) for name, path in MODEL_PATHS.items()}
print('All models loaded. Check http://127.0.0.1:5000/')

# You must update these names to match the last conv layer in your models!
LAST_CONV_LAYER = {
    "keras": "conv2d_25",  # CNN - Grad-CAM will NOT be generated for this
    "keras1": "separable_conv2d_1",  # Separable CNN - Grad-CAM will NOT be generated for this
    
    "keras2": "block14_sepconv2"  # Xception default - Grad-CAM WILL be generated for this
}


def preprocess_image(img, model_name):
    if model_name in ['keras', 'keras1']:
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((64, 64))
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=-1)  # Add channel dimension
        x = np.expand_dims(x, axis=0)  # Add batch dimension
    elif model_name == 'keras2':  # Xception
        img = img.convert('RGB')  # Convert to RGB
        img = img.resize((299, 299))
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)  # Add batch dimension
    else:
        raise ValueError("Unknown model name")
    return x


def model_predict(img, model_name, model):
    x = preprocess_image(img, model_name)
    preds = model.predict(x)
    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    pred_proba = float(np.max(preds))
    return pred_class, pred_proba, pred_idx, x


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:  # Should not happen if model_predict is called first
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)  # Normalize
    return heatmap.numpy()


def overlay_heatmap(heatmap, pil_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Convert PIL image to OpenCV format
    img = np.array(pil_img)
    if img.ndim == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    # else: assume RGB, no conversion needed if it's already BGR for OpenCV

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Superimpose the heatmap
    superimposed_img_bgr = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    # Convert back to RGB for PIL
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_img_rgb)


def pil_image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")  # Save as PNG
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/auth.html', methods=['GET'])
def auth():
    return render_template('auth.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    model_choice = request.form.get('model_choice')

    if model_choice not in models:
        return jsonify({'error': 'Invalid model choice'}), 400

    img = Image.open(file.stream)
    pred_class, pred_proba, pred_idx, x = model_predict(img, model_choice, models[model_choice])
    explanation = disease_explanations.get(pred_class, {"info": "No information available for this condition.",
                                                        "curability": "Unknown"})

    # Grad-CAM: For CNN ("keras") and Xception ("keras2")
    if model_choice in ['keras', 'keras2', 'keras1']:
        last_conv = LAST_CONV_LAYER[model_choice]
        heatmap = make_gradcam_heatmap(x, models[model_choice], last_conv, pred_idx)

        # Overlay heatmap on input image (resize original to match model input)
        if model_choice == 'keras2':
            vis_img = img.convert('RGB').resize((299, 299))
        else:  # For CNN, input is grayscale and resized to (64, 64)
            vis_img = img.convert('L').resize((299, 299))

        gradcam_img = overlay_heatmap(heatmap, vis_img)

        # ADD NEW FEATURES HERE
        color_percentages = get_color_percentages(gradcam_img)
        try:
            ai_explanation = analyze_gradcam_smart(gradcam_img, pred_class, color_percentages)
        except Exception as e:
            ai_explanation = "AI analysis temporarily unavailable"

        gradcam_b64 = pil_image_to_base64(gradcam_img)
    else:
        gradcam_b64 = None
        color_percentages = None
        ai_explanation = "AI analysis not available for this model"

    return jsonify(
        result=pred_class,
        probability=round(pred_proba, 3),
        disease_info=explanation["info"],
        curability=explanation["curability"],
        gradcam_image=gradcam_b64,
        color_percentages=color_percentages,  # ADD THIS
        ai_analysis=ai_explanation  # ADD THIS
    )


if __name__ == '__main__':
    # Make sure 'templates' directory exists and 'index.html' is inside it.
    # Make sure 'models' directory exists with your .keras model files.
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()