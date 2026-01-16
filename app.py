from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# ===== ЗАГРУЗКА МОДЕЛЕЙ =====
acne_model = tf.keras.models.load_model("acne_model.h5")
eyes_model = tf.keras.models.load_model("eyes_model.h5")
nails_model = tf.keras.models.load_model("nails_model.h5")
xray_model = tf.keras.models.load_model("xray_model.h5")

# ===== КЛАССЫ РЕНТГЕНА =====
xray_classes = [
    "Abscess",
    "ARDS",
    "Atelectasis",
    "Atherosclerosis of the aorta",
    "Cardiomegaly",
    "Emphysema",
    "Fracture",
    "Hydropneumothorax",
    "Hydrothorax",
    "Pneumonia",
    "Pneumosclerosis",
    "Post inflammatory changes",
    "Post traumatic ribs deformation",
    "Sarcoidosis",
    "Scoliosis",
    "Tuberculosis",
    "Venous congestion"
]

# ===== ПОДГОТОВКА ИЗОБРАЖЕНИЯ =====
def prepare_image(file):
    img = Image.open(file).convert("RGB")
    img_resized = img.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return arr, img_base64


@app.route("/", methods=["GET", "POST"])
def index():
    skin_result = ""
    eyes_result = ""
    nails_result = ""
    xray_result = ""

    skin_img = None
    eyes_img = None
    nails_img = None
    xray_img = None

    if request.method == "POST":

        # ===== КОЖА =====
        if "skin_image" in request.files and request.files["skin_image"].filename != "":
            arr, skin_img = prepare_image(request.files["skin_image"])
            pred = acne_model.predict(arr)[0][0]

            skin_result = (
                "Обнаружены визуальные признаки акне"
                if pred < 0.5 else
                "Визуальные признаки акне не обнаружены"
            )

        # ===== ГЛАЗА =====
        if "eyes_image" in request.files and request.files["eyes_image"].filename != "":
            arr, eyes_img = prepare_image(request.files["eyes_image"])
            pred = eyes_model.predict(arr)[0][0]

            eyes_result = (
                "Возможные визуальные признаки воспаления"
                if pred > 0.5 else
                "Глаз выглядит здоровым"
            )

        # ===== НОГТИ =====
        if "nails_image" in request.files and request.files["nails_image"].filename != "":
            arr, nails_img = prepare_image(request.files["nails_image"])
            pred = nails_model.predict(arr)[0][0]

            if pred >= 0.5:
                nails_result = f"Обнаружены визуальные признаки нездорового ногтя ({pred*100:.1f}%)"
            else:
                nails_result = f"Ноготь выглядит здоровым ({(1-pred)*100:.1f}%)"

        # ===== РЕНТГЕН =====
        if "xray_image" in request.files and request.files["xray_image"].filename != "":
            arr, xray_img = prepare_image(request.files["xray_image"])

            preds = xray_model.predict(arr)[0]
            index = np.argmax(preds)
            confidence = preds[index] * 100
            label = xray_classes[index]

            xray_result = f"{label} ({confidence:.1f}%)"

    return render_template(
        "index.html",
        skin_result=skin_result,
        eyes_result=eyes_result,
        nails_result=nails_result,
        xray_result=xray_result,
        skin_img=skin_img,
        eyes_img=eyes_img,
        nails_img=nails_img,
        xray_img=xray_img
    )


if __name__ == "__main__":
    app.run(debug=True)
