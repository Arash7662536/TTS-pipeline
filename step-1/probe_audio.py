"""Identify the audio container in an Arrow dataset and, for WAV, get duration."""
import io, sys, wave
import pyarrow as pa
from persian_tts_frontend.cli import arrow_fragments, _arrow_batches

MAGIC = {b"RIFF": "wav", b"fLaC": "flac", b"OggS": "ogg",
         b"\xff\xfb": "mp3", b"\xff\xf3": "mp3", b"\xff\xf2": "mp3",
         b"ID3": "mp3", b"\x00\x00\x00 ftyp": "m4a"}

def sniff(b):
    for m, name in MAGIC.items():
        if b.startswith(m):
            return name
    return "unknown:" + b[:8].hex()

def wav_seconds(b):
    with wave.open(io.BytesIO(b)) as w:
        return w.getnframes() / float(w.getframerate())

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
files = arrow_fragments(path)
shown = 0
for batch in _arrow_batches(pa, files[0]):
    col = batch.column(batch.schema.names.index("audio"))
    blobs = col.field("bytes")
    for i in range(batch.num_rows):
        if shown >= limit:
            sys.exit(0)
        b = blobs[i].as_py()
        kind = sniff(b)
        extra = ""
        if kind == "wav":
            try:
                with wave.open(io.BytesIO(b)) as w:
                    extra = (f"  {w.getframerate()} Hz, {w.getnchannels()} ch, "
                             f"{w.getsampwidth()*8}-bit, {wav_seconds(b):.2f} s")
            except Exception as e:
                extra = f"  (unreadable: {e})"
        print(f"row {shown}: {len(b)} bytes  {kind}{extra}")
        shown += 1
