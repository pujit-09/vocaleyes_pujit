import os
import re
import base64
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template,
    abort,
)
from flask_cors import CORS
from gtts import gTTS
from PIL import Image

# ----------------------------
# OCR Tamil Model
# ----------------------------
from ocr_tamil.ocr import OCR

# Initialize Tamil + English OCR
ocr_model = OCR(detect=True)

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def unique_filename(prefix="file", ext="png"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.{ext}"

def pil_from_bytes(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")

    # Preprocess draw images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
    )
    binary = cv2.medianBlur(binary, 3)
    img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img)

# ----------------------------
# OCR Prediction
# ----------------------------
def predict_text_from_bytes(image_bytes):
    try:
        temp_path = UPLOAD_FOLDER / unique_filename("temp", "jpg")
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        text_list = ocr_model.predict(str(temp_path))
        temp_path.unlink(missing_ok=True)

        if isinstance(text_list, list):
            if isinstance(text_list[0], list):
                flat_text = " ".join([" ".join(t) for t in text_list])
            else:
                flat_text = " ".join(text_list)
        else:
            flat_text = str(text_list)
        return flat_text.strip()
    except Exception as e:
        print("⚠️ OCR Tamil error:", e)
        return ""

# ----------------------------
# Text-to-Speech
# ----------------------------
def choose_tts_lang_for_text(text):
    tamil_chars = re.findall(r"[\u0B80-\u0BFF]", text)
    if len(tamil_chars) > 3:
        return "ta"
    return "en"

def create_tts(text, audio_path):
    if not text or not text.strip():
        return False
    try:
        lang = choose_tts_lang_for_text(text)
        tts = gTTS(text=text, lang=lang)
        tts.save(str(audio_path))
        return True
    except Exception as e:
        print("⚠️ gTTS error:", e)
        return False

# ----------------------------
# Page Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/draw")
def draw_page():
    return render_template("draw.html")

@app.route("/upload-page")
def upload_page():
    return render_template("upload.html")

@app.route("/capture")
def capture_page():
    return render_template("capture.html")

@app.route("/contact")
def contact_page():
    return render_template("contact.html")

# 🆕 VoicePad Routes
@app.route("/voicepad")
def voicepad_page():
    return render_template("voicepad.html")

@app.route("/speak", methods=["POST"])
def speak_api():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing text"}), 400

    text = data["text"].strip()
    lang = data.get("lang", "en-IN")

    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        # Detect gTTS language
        if lang.startswith("ta"):
            gtts_lang = "ta"
        elif lang.startswith("hi"):
            gtts_lang = "hi"
        else:
            gtts_lang = "en"

        audio_name = unique_filename("voicepad", "mp3")
        audio_path = UPLOAD_FOLDER / audio_name
        tts = gTTS(text=text, lang=gtts_lang)
        tts.save(str(audio_path))

        return jsonify({"audio_url": f"/play-audio/{audio_name}"})
    except Exception as e:
        print("⚠️ /speak error:", e)
        return jsonify({"error": "Server error"}), 500

# ----------------------------
# Upload API
# ----------------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    safe_name = unique_filename("upload", file.filename.rsplit(".", 1)[-1])
    file_path = UPLOAD_FOLDER / safe_name
    file.save(str(file_path))

    try:
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        text = predict_text_from_bytes(img_bytes)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500

    audio_name = unique_filename("voice", "mp3")
    audio_path = UPLOAD_FOLDER / audio_name
    create_tts(text, audio_path)
    file_path.unlink(missing_ok=True)

    return jsonify({
        "filename": file.filename,
        "extracted_text": text,
        "audio_play_url": f"/play-audio/{audio_name}",
        "audio_download_url": f"/download-audio/{audio_name}",
    })

@app.route("/api/upload", methods=["POST"])
def upload_api():
    return upload_file()

@app.route("/process", methods=["POST"])
def process_file():
    if "file" not in request.files:
        return render_template("upload.html", error="No file uploaded")
    file = request.files["file"]
    if file.filename == "":
        return render_template("upload.html", error="Empty filename")

    safe_name = unique_filename("upload", file.filename.rsplit(".", 1)[-1])
    file_path = UPLOAD_FOLDER / safe_name
    file.save(str(file_path))

    try:
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        text = predict_text_from_bytes(img_bytes)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        return render_template("upload.html", error=str(e))

    audio_name = unique_filename("voice", "mp3")
    audio_path = UPLOAD_FOLDER / audio_name
    create_tts(text, audio_path)
    file_path.unlink(missing_ok=True)

    return render_template(
        "upload.html",
        filename=file.filename,
        extracted_text=text,
        audio_play_url=f"/play-audio/{audio_name}",
        audio_download_url=f"/download-audio/{audio_name}",
    )

# ----------------------------
# Draw API
# ----------------------------
@app.route("/api/draw", methods=["POST"])
def draw_api():
    data = request.get_json()
    if not data or "dataURL" not in data:
        return jsonify({"error": "Missing dataURL"}), 400

    m = re.match(r"data:image/[^;]+;base64,(.*)", data["dataURL"])
    if not m:
        return jsonify({"error": "Invalid dataURL"}), 400
    try:
        img_bytes = base64.b64decode(m.group(1))
    except Exception:
        return jsonify({"error": "Base64 decode failed"}), 400

    image_name = unique_filename("draw", "png")
    image_path = UPLOAD_FOLDER / image_name
    with open(image_path, "wb") as f:
        f.write(img_bytes)

    try:
        text = predict_text_from_bytes(img_bytes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    audio_name = unique_filename("voice", "mp3")
    audio_path = UPLOAD_FOLDER / audio_name
    create_tts(text, audio_path)

    return jsonify({
        "saved_image_url": f"/uploads/{image_name}",
        "extracted_text": text,
        "audio_play_url": f"/play-audio/{audio_name}",
        "audio_download_url": f"/download-audio/{audio_name}",
    })

# ----------------------------
# Capture API
# ----------------------------
@app.route("/api/capture-upload", methods=["POST"])
def capture_upload_api():
    return upload_file()

# ----------------------------
# File serving
# ----------------------------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    p = UPLOAD_FOLDER / filename
    return send_file(str(p)) if p.exists() else abort(404)

@app.route("/play-audio/<audio_name>")
def play_audio(audio_name):
    p = UPLOAD_FOLDER / audio_name
    return send_file(str(p), mimetype="audio/mpeg", as_attachment=False) if p.exists() else abort(404)

@app.route("/download-audio/<audio_name>")
def download_audio(audio_name):
    p = UPLOAD_FOLDER / audio_name
    return send_file(str(p), mimetype="audio/mpeg", as_attachment=True, download_name=audio_name) if p.exists() else abort(404)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    print("🚀 Running at: http://127.0.0.1:5000/")
    app.run(debug=True, host="0.0.0.0", port=5000)
