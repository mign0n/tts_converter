from pathlib import Path

import scipy
from torch import Tensor

DEFAULT_SAMPLE_RATE = 16000


def save_audio(
    path: Path,
    audio: Tensor,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    scipy.io.wavfile.write(
        path,
        rate=sample_rate,
        data=audio.numpy().squeeze(),
    )
