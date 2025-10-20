# app.py
import base64
import io
import json
from datetime import datetime

from flask import Flask, request, jsonify
from google.cloud import firestore

import numpy as np
import face_recognition

app = Flask(__name__)

db = firestore.Client()
FACES_COLLECTION = "faces"
TOLERANCE = 0.6  # adjust as needed

def decode_base64_image(b64_string):
    image_data = base64.b64decode(b64_string)
    image = face_recognition.load_image_file(io.BytesIO(image_data))
    return image

@app.route("/scan_face", methods=["POST"])
def scan_face():
    """
    This endpoint receives:
      {
        "uid": "<user-id>",
        "image_base64": "<base64 string>"
      }
    It will detect a single face in the image, encode it, and store it in Firestore under the provided uid.
    """
    data = request.get_json()
    uid = data.get("uid")
    image_b64 = data.get("image_base64")
    if not uid or not image_b64:
        return jsonify({"error": "Both 'uid' and 'image_base64' are required"}), 400

    try:
        image = decode_base64_image(image_b64)
    except Exception as e:
        return jsonify({"error": "Bad image data", "details": str(e)}), 400

    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        return jsonify({"error": "Expected exactly one face in the image, found {}".format(len(face_locations))}), 400

    face_encodings = face_recognition.face_encodings(image, face_locations)
    encoding = face_encodings[0]
    encoding_list = encoding.tolist()

    # Store in Firestore
    doc_ref = db.collection(FACES_COLLECTION).document(uid)
    doc_ref.set({
        "uid": uid,
        "encoding": encoding_list,
        "registered_at": datetime.utcnow().isoformat()
    }, merge=True)

    return jsonify({"status": "ok", "uid": uid}), 200

@app.route("/recognize_multi_faces", methods=["POST"])
def recognize_multi_faces():
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
