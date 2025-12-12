# app.py
import base64
import io
import json
from datetime import datetime

from flask import Flask, request, jsonify
from google.cloud import firestore
from flask_cors import CORS

import numpy as np
import face_recognition
from PIL import Image
from google.oauth2 import service_account

app = Flask(__name__)
CORS(app)

creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
creds = service_account.Credentials.from_service_account_info(json.loads(creds_json))

db = firestore.Client(credentials=creds, project=creds.project_id)
FACES_COLLECTION = "faces"
TOLERANCE = 0.6  # adjust as needed

def decode_base64_image(b64_string):
    try:
        image_data = base64.b64decode(b64_string)
        pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")  # force RGB
        return np.array(pil_img)
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

@app.route("/scan_face", methods=["POST"])
def scan_face():
    data = request.get_json()
    uid = data.get("uid")
    image_b64 = data.get("image_base64")

    if not uid or not image_b64:
        return jsonify({"error": "Both 'uid' and 'image_base64' are required"}), 400

    try:
        image = decode_base64_image(image_b64)
    except Exception as e:
        return jsonify({"error": "Bad image data", "details": str(e)}), 400

    # CNN for perfect detection
    face_locations = face_recognition.face_locations(image, model="cnn")

    if len(face_locations) != 1:
        return jsonify({
            "error": f"Expected exactly one face, found {len(face_locations)}"
        }), 400

    face_encodings = face_recognition.face_encodings(image, face_locations)
    encoding = face_encodings[0].tolist()

    db.collection(FACES_COLLECTION).document(uid).set({
        "uid": uid,
        "encoding": encoding,
        "registered_at": datetime.utcnow().isoformat()
    }, merge=True)

    return jsonify({"status": "ok", "uid": uid}), 200

@app.route("/recognize_multi_faces_old", methods=["POST"])
def recognize_multi_facesOld():
    """
    This endpoint receives:
      {
        "image_base64": "<base64 string>"
      }
    It will detect potentially multiple faces in the image, encode each, compare against stored known faces in Firestore,
    and return JSON with recognized uids (or null) + distances.
    """
    data = request.get_json()
    image_b64 = data.get("image_base64")
    if not image_b64:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        image = decode_base64_image(image_b64)
    except Exception as e:
        return jsonify({"error": "Bad image data", "details": str(e)}), 400

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # Load known faces from Firestore
    docs = db.collection(FACES_COLLECTION).stream()
    known_uids = []
    known_encodings = []
    for doc in docs:
        rec = doc.to_dict()
        # assume encoding field is present
        known_uids.append(rec["uid"])
        known_encodings.append(np.array(rec["encoding"], dtype=float))

    results = []
    for loc, encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, encoding) if known_encodings else []
        best_uid = None
        best_dist = None
        if len(distances) > 0:
            idx = int(np.argmin(distances))
            best_dist = float(distances[idx])
            if best_dist < TOLERANCE:
                best_uid = known_uids[idx]
        results.append({
            "bbox": [loc[0], loc[1], loc[2], loc[3]],
            "uid": best_uid,
            "distance": best_dist
        })

    return jsonify({"faces": results}), 200
@app.route("/recognize_multi_faces", methods=["POST"])
def recognize_multi_faces():
    data = request.get_json()
    image_b64 = data.get("image_base64")

    if not image_b64:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        image = decode_base64_image(image_b64)
    except Exception as e:
        return jsonify({"error": "Bad image data", "details": str(e)}), 400

    # CNN = better accuracy for multiple faces
    face_locations = face_recognition.face_locations(image, model="cnn")
    face_encodings = face_recognition.face_encodings(image, face_locations)

    docs = db.collection(FACES_COLLECTION).stream()
    known_uids = []
    known_encodings = []

    for doc in docs:
        rec = doc.to_dict()
        if rec and "uid" in rec and "encoding" in rec:
            known_uids.append(rec["uid"])
            known_encodings.append(np.array(rec["encoding"], dtype=float))

    if len(known_encodings) == 0:
        return jsonify({"error": "No known faces found. Register faces first."}), 400

    results = []

    for loc, encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, encoding)
        idx = int(np.argmin(distances))
        best_dist = float(distances[idx])
        best_uid = known_uids[idx] if best_dist < TOLERANCE else None

        results.append({
            "bbox": [int(x) for x in loc],
            "uid": best_uid,
            "distance": round(best_dist, 4)
        })

    return jsonify({
        "status": "ok",
        "faces": results,
        "count": len(results)
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

