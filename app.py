from flask import Flask, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import mediapipe as mp

# ------------------------------
# CONFIG
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXT = (".jpg", ".jpeg", ".png")

app = Flask(__name__)

# MediaPipe & ORB
mp_hands = mp.solutions.hands
orb = cv2.ORB_create(nfeatures=1000)


# ------------------------------
# HELPERS
# ------------------------------
def allowed_file(filename):
    return filename.lower().endswith(ALLOWED_EXT)


def detect_and_crop_hand(image):
    """
    Return cropped hand image, landmarks, and confidence
    """
    try:
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                return None, None, 0.0

            h, w = image.shape[:2]
            hand_landmarks = results.multi_hand_landmarks[0]

            x_min, y_min = w, h
            x_max, y_max = 0, 0
            landmarks = []

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            padding = int(min(w, h) * 0.05)  # 5% padding
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            if x_max - x_min <= 0 or y_max - y_min <= 0:
                return None, None, 0.0

            cropped = image[y_min:y_max, x_min:x_max]
            confidence = results.multi_handedness[0].classification[0].score
            return cropped, landmarks, confidence
    except Exception:
        app.logger.exception("Error in detect_and_crop_hand:")
        return None, None, 0.0


def extract_descriptors(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return des


def identify_owner_by_orb(cropped_img):
    """
    Simple ORB matching: return (best_owner_name or None, best_score_int)
    """
    if not os.path.isdir(DATASET_FOLDER):
        return None, 0

    query_des = extract_descriptors(cropped_img)
    if query_des is None:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_owner = None
    best_score = 0

    for person in os.listdir(DATASET_FOLDER):
        person_dir = os.path.join(DATASET_FOLDER, person)
        if not os.path.isdir(person_dir):
            continue

        for fname in os.listdir(person_dir):
            if not allowed_file(fname):
                continue
            ppath = os.path.join(person_dir, fname)
            train_img = cv2.imread(ppath)
            if train_img is None:
                continue
            train_des = extract_descriptors(train_img)
            if train_des is None:
                continue

            try:
                matches = bf.match(query_des, train_des)
                score = len(matches)
            except Exception:
                score = 0

            if score > best_score:
                best_score = score
                best_owner = person

    return best_owner, int(best_score)


# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "PalmServer aktif",
        "endpoints": {
            "identify_json": "/api/identify/visual (POST file)",
            "processed_file": "/processed/<filename>",
            "list_dataset": "/api/list"
        }
    })


@app.route("/api/list", methods=["GET"])
def api_list():
    if not os.path.exists(DATASET_FOLDER):
        return jsonify({"success": False, "message": f"Dataset folder not found: {DATASET_FOLDER}"}), 404
    owners = [d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))]
    return jsonify({"success": True, "dataset_owners": owners, "dataset_path": DATASET_FOLDER})


@app.route("/processed/<path:filename>", methods=["GET"])
def get_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=False)


@app.route("/api/identify/visual", methods=["POST"])
def identify_visual():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    filename_safe = secure_filename(file.filename)
    if not allowed_file(filename_safe):
        return jsonify({"success": False, "message": "File extension not allowed"}), 400

    upload_path = os.path.join(UPLOAD_FOLDER, filename_safe)
    file.save(upload_path)

    img = cv2.imread(upload_path)
    if img is None:
        return jsonify({"success": False, "message": "Could not read saved image"}), 400

    cropped, landmarks, confidence = detect_and_crop_hand(img)
    if cropped is None:
        return jsonify({"success": False, "message": "No hand detected"}), 200

    processed_name = f"cropped_{filename_safe}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_name)
    cv2.imwrite(processed_path, cropped)

    owner, score = identify_owner_by_orb(cropped)
    if owner is None:
        owner = "Tidak dikenali"

    image_url = url_for("get_processed_file", filename=processed_name, _external=True)

    return jsonify({
        "success": True,
        "owner": owner,
        "score": score,
        "confidence": confidence,
        "landmarks": landmarks,  # koordinat tangan
        "image_url": image_url,
        "processed_path": processed_path
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    # debug=False untuk deploy (jaga keamanan)
    app.run(host="0.0.0.0", port=port, debug=False)

