from flask import Flask, request, jsonify, send_file
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

# ===============================
# Direktori & folder
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
OWNER_NAMES = ["Subhan", "Hatim", "Romlah", "Rozi"]

# ===============================
# Fungsi util
# ===============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_image(img, size=(300, 300)):
    """Resize dan konversi ke grayscale"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, size)
    return resized

def load_dataset():
    dataset = []
    for i in range(1, 5):
        filename = f"palm{i}"
        for ext in ['', '.jpg', '.jpeg', '.png', '.bmp']:
            test_path = os.path.join(DATASET_FOLDER, filename + ext)
            if os.path.exists(test_path):
                img = cv2.imread(test_path)
                if img is not None:
                    img_norm = normalize_image(img)
                    owner = OWNER_NAMES[i-1] if i-1 < len(OWNER_NAMES) else "Tidak diketahui"
                    dataset.append((os.path.basename(test_path), img_norm, owner))
                    break
    return dataset

dataset = load_dataset()
orb = cv2.ORB_create(nfeatures=2000)

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return jsonify({
        "message": "Palm Identification API with Visuals",
        "status": "active",
        "dataset_count": len(dataset),
        "registered_owners": OWNER_NAMES
    })

@app.route("/api/identify", methods=["POST"])
def identify():
    if len(dataset) == 0:
        return jsonify({"success": False, "error": "Dataset kosong."}), 400
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file diunggah"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File tidak valid"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img_test = cv2.imread(filepath)
    if img_test is None:
        return jsonify({"success": False, "error": "File tidak bisa dibaca."}), 400

    img_test = normalize_image(img_test)

    kp2, des2 = orb.detectAndCompute(img_test, None)
    if des2 is None:
        return jsonify({"success": False, "error": "Tidak dapat mengekstrak fitur."}), 400

    best_score = float('inf')
    best_owner = "Tidak Dikenali"
    match_details = []

    for name, img_dataset, owner in dataset:
        kp1, des1 = orb.detectAndCompute(img_dataset, None)
        if des1 is None:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if matches:
            score = sum([m.distance for m in matches]) / len(matches)
            match_count = len(matches)
        else:
            score = 1000
            match_count = 0

        match_details.append({
            "dataset_file": name,
            "owner": owner,
            "score": float(score),
            "matches": match_count
        })

        if score < best_score:
            best_score = score
            best_owner = owner

    if best_score < 40:
        confidence = "Tinggi"
    elif best_score < 60:
        confidence = "Sedang"
    else:
        confidence = "Rendah"

    return jsonify({
        "success": True,
        "identified_owner": best_owner,
        "confidence_score": float(best_score),
        "confidence_level": confidence,
        "match_details": match_details
    })

@app.route("/api/identify/visual", methods=["POST"])
def identify_visual():
    if len(dataset) == 0:
        return jsonify({"success": False, "error": "Dataset kosong."}), 400
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file diunggah"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File tidak valid"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img_test = cv2.imread(filepath)
    img_test = normalize_image(img_test)

    kp2, des2 = orb.detectAndCompute(img_test, None)
    if des2 is None:
        return jsonify({"success": False, "error": "Tidak dapat mengekstrak fitur."}), 400

    best_score = float('inf')
    best_owner = "Tidak Dikenali"
    best_image = None

    for name, img_dataset, owner in dataset:
        kp1, des1 = orb.detectAndCompute(img_dataset, None)
        if des1 is None:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if matches:
            score = sum([m.distance for m in matches]) / len(matches)
            if score < best_score:
                best_score = score
                best_owner = owner
                best_image = img_dataset

    # Tentukan confidence
    if best_score < 40:
        confidence = "Tinggi"
    elif best_score < 60:
        confidence = "Sedang"
    else:
        confidence = "Rendah"

    # Mode JSON
    if request.args.get("mode") == "json":
        return jsonify({
            "success": True,
            "identified_owner": best_owner,
            "confidence_score": float(best_score),
            "confidence_level": confidence
        })

    # Mode visual
    if best_image is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp1, des1 = orb.detectAndCompute(best_image, None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img_matches = cv2.drawMatches(best_image, kp1, img_test, kp2, matches[:20], None, flags=2)

        _, buffer = cv2.imencode('.png', img_matches)
        io_buf = BytesIO(buffer)
        return send_file(io_buf, mimetype='image/png')

    else:
        return jsonify({
            "success": True,
            "identified_owner": best_owner,
            "confidence_level": "Rendah",
            "message": "Tidak ada dataset cocok ditemukan."
        })

# ===============================
# Jalankan server
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
