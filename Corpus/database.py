from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class AudioMetadata(db.Model):
    __tablename__ = "audio_metadata"

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255))
    absolute_path = db.Column(db.String(1024))
    relative_path = db.Column(db.String(1024))
    genre = db.Column(db.String(255))
    file_extension = db.Column(db.String(20))
    file_size_bytes = db.Column(db.Integer)
    compressed_size_bytes = db.Column(db.Integer)
    compression_ratio = db.Column(db.Float)
    sha256_hash = db.Column(db.String(128))
    sample_rate = db.Column(db.Integer)
    channels = db.Column(db.Integer)
    subtype = db.Column(db.String(100))
    duration_seconds = db.Column(db.Float)
    bitrate = db.Column(db.Integer)
    length = db.Column(db.Float)
    sample_rate_mutagen = db.Column(db.Integer)
    error_soundfile = db.Column(db.Text)
    error_mutagen = db.Column(db.Text)
    source_name = db.Column(db.String(255))
    source_url = db.Column(db.String(512))
    source_type = db.Column(db.String(100))
    indexed_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # NEW

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class ChangeLog(db.Model):
    __tablename__ = "change_logs"

    id = db.Column(db.Integer, primary_key=True)
    table_name = db.Column(db.String(255))
    record_id = db.Column(db.Integer)
    action = db.Column(db.String(50))  # INSERT, UPDATE, DELETE
    field_name = db.Column(db.String(255), nullable=True)
    old_value = db.Column(db.Text, nullable=True)
    new_value = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.Column(db.String(255), default="system")

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
