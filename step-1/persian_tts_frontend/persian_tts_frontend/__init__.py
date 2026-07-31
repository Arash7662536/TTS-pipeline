"""
Persian TTS frontend for VoxCPM 2 fine-tuning.

Step 1 of the build plan. This package is the frozen artifact that must be
byte-identical at training time and at serving time.

Quick start
-----------
    from persian_tts_frontend import Normalizer

    nz = Normalizer()
    print(nz.version)                       # record this in every manifest row
    print(nz("در سال ۱۴۰۳ حدود ٪۲۵ رشد داشت."))

Dataloader-side masking (Stage A)
---------------------------------
    from persian_tts_frontend import MaskingSchedule

    sched = MaskingSchedule(total_steps=8000)
    text = sched.apply(row["text"], step=global_step, priority=HOMOGRAPHS)

Tokenizer audit -- run before Stage A
-------------------------------------
    from persian_tts_frontend import audit
    audit.print_report(audit.roundtrip(tok), "roundtrip")
    audit.print_report(audit.fertility(tok, texts), "fertility")
    audit.print_report(audit.survival(tok, rows, tokens_per_second=TPS),
                       "clip survival")
"""

from . import audit, chars, diacritics, expand, latin, normalize, numbers
from .diacritics import MaskingSchedule, mask, sample_density, strip
from .latin import LatinResolver
from .pipeline import Normalizer, NormalizerConfig, NormResult

__version__ = "1.0.0"

__all__ = [
    "Normalizer", "NormalizerConfig", "NormResult",
    "MaskingSchedule", "mask", "sample_density", "strip",
    "LatinResolver",
    "audit", "chars", "diacritics", "expand", "latin", "normalize", "numbers",
]
