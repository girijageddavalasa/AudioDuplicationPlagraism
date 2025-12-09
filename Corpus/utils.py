import os
import gzip
import hashlib
import soundfile as sf
from mutagen import File as MutagenFile
from database import db, ChangeLog
from datetime import datetime

# ------------------- LOGGING -------------------
def log_change(table_name, record_id, action, field_name=None, old_value=None, new_value=None, user="system"):
    log_entry = ChangeLog(
        table_name=table_name,
        record_id=record_id,
        action=action,
        field_name=field_name,
        old_value=str(old_value) if old_value else None,
        new_value=str(new_value) if new_value else None,
        user=user,
        timestamp=datetime.utcnow()
    )
    db.session.add(log_entry)
    db.session.commit()


# ------------------- AUDIO METADATA HELPERS -------------------
def compute_compressed_size(file_path):
    try:
        with open(file_path, 'rb') as f:
            return len(gzip.compress(f.read()))
    except Exception:
        return None


def compute_file_hash(file_path):
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def extract_audio_metadata(file_path):
    meta = {}
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        with sf.SoundFile(file_path) as f:
            duration = len(f) / f.samplerate
            bit_depth = 16  # Default; WAV files often 16-bit PCM
            bitrate_calc = f.samplerate * bit_depth * f.channels

            meta.update({
                "sample_rate": f.samplerate,
                "channels": f.channels,
                "subtype": f.subtype,
                "duration_seconds": round(duration, 3),
                "bitrate": bitrate_calc,
                "length": round(duration, 3),
                "sample_rate_mutagen": f.samplerate,
                "error_soundfile": None,
                "error_mutagen": None
            })
    except Exception as e:
        meta.update({
            "sample_rate": None,
            "channels": None,
            "subtype": None,
            "duration_seconds": None,
            "bitrate": None,
            "length": None,
            "sample_rate_mutagen": None,
            "error_soundfile": str(e),
            "error_mutagen": None
        })

    # Mutagen for compressed formats only
    if file_ext in {".mp3", ".flac", ".ogg", ".au"}:
        try:
            mf = MutagenFile(file_path, easy=True)
            if mf and mf.info:
                meta.update({
                    "bitrate": getattr(mf.info, "bitrate", meta["bitrate"]),
                    "length": getattr(mf.info, "length", meta["length"]),
                    "sample_rate_mutagen": getattr(mf.info, "sample_rate", meta["sample_rate_mutagen"])
                })
        except Exception as e:
            meta["error_mutagen"] = str(e)

    return meta


def gather_audio_metadata(file_path, source_name=None, source_url=None, source_type=None):
    file_size = os.path.getsize(file_path)
    compressed_size = compute_compressed_size(file_path)
    compression_ratio = round(compressed_size / file_size, 4) if compressed_size else None
    sha_hash = compute_file_hash(file_path)
    meta = extract_audio_metadata(file_path)

    return {
        "file_name": os.path.basename(file_path),
        "absolute_path": os.path.abspath(file_path),
        "relative_path": os.path.relpath(file_path),
        "genre": os.path.basename(os.path.dirname(file_path)),
        "file_extension": os.path.splitext(file_path)[1].lower(),
        "file_size_bytes": file_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": compression_ratio,
        "sha256_hash": sha_hash,
        **meta,
        "source_name": source_name,
        "source_url": source_url,
        "source_type": source_type,
        "last_updated_at": datetime.utcnow()  # automatically populated
    }
