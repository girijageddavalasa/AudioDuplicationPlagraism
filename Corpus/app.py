from flask import Flask, request, jsonify
from flask_cors import CORS
from database import db, AudioMetadata, ChangeLog
from utils import gather_audio_metadata, log_change
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- Database config ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///audio_metadata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route("/")
def index():
    return jsonify({"message": "Audio Metadata API is running!"})

# --- Add Single File ---
@app.route("/add_file", methods=["POST"])
def add_file():
    """
    POST JSON:
    {
      "file_path": "C:/path/to/audio.wav",
      "source_name": "GTZAN Dataset",
      "source_url": "https://www.kaggle.com/dataset/...",
      "source_type": "kaggle"
    }
    """
    data = request.get_json()
    file_path = data.get("file_path")

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File path is invalid or does not exist"}), 400

    # Collect metadata
    meta = gather_audio_metadata(
        file_path=file_path,
        source_name=data.get("source_name"),
        source_url=data.get("source_url"),
        source_type=data.get("source_type")
    )

    # Add record to DB
    record = AudioMetadata(**meta)
    db.session.add(record)
    db.session.commit()

    # Log this insert
    log_change("audio_metadata", record.id, "INSERT", user="API")

    return jsonify({"message": "File metadata added", "metadata": record.to_dict()}), 201


# --- Bulk Scan Directory ---
@app.route("/scan_directory", methods=["POST"])
def scan_directory():
    """
    POST JSON:
    {
      "directory_path": "C:/path/to/dataset",
      "source_name": "GTZAN Dataset",
      "source_url": "...",
      "source_type": "kaggle"
    }
    """
    data = request.get_json()
    directory_path = data.get("directory_path")

    if not directory_path or not os.path.isdir(directory_path):
        return jsonify({"error": "Invalid directory path"}), 400

    count = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if not file.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".au")):
                continue

            file_path = os.path.join(root, file)
            meta = gather_audio_metadata(
                file_path=file_path,
                source_name=data.get("source_name"),
                source_url=data.get("source_url"),
                source_type=data.get("source_type")
            )

            record = AudioMetadata(**meta)
            db.session.add(record)
            db.session.commit()
            log_change("audio_metadata", record.id, "INSERT", user="API")
            count += 1

    return jsonify({"message": f"Scanned {count} files successfully."}), 201


# --- Get all metadata ---
@app.route("/get_all", methods=["GET"])
def get_all():
    all_data = AudioMetadata.query.all()
    return jsonify([a.to_dict() for a in all_data])


# --- Get by filename ---
@app.route("/get_by_name/<string:file_name>", methods=["GET"])
def get_by_name(file_name):
    record = AudioMetadata.query.filter_by(file_name=file_name).first()
    if record:
        return jsonify(record.to_dict())
    return jsonify({"error": "File not found"}), 404


# --- Update existing record ---
@app.route("/update_file/<int:file_id>", methods=["PUT"])
def update_file(file_id):
    data = request.get_json()
    record = AudioMetadata.query.get(file_id)
    if not record:
        return jsonify({"error": "File not found"}), 404

    for key, value in data.items():
        if hasattr(record, key):
            old_value = getattr(record, key)
            setattr(record, key, value)
            log_change("audio_metadata", file_id, "UPDATE", field_name=key, old_value=old_value, new_value=value, user="API")

    record.last_updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"message": "File updated successfully", "metadata": record.to_dict()})


# --- Fetch all change logs ---
@app.route("/logs", methods=["GET"])
def get_logs():
    logs = ChangeLog.query.order_by(ChangeLog.timestamp.desc()).all()
    return jsonify([l.to_dict() for l in logs])


if __name__ == "__main__":
    app.run(debug=True)
