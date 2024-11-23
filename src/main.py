from pathlib import Path

from utils import save_audio
from tts import TextToAudio

INPUT_MEDIA = Path(__file__).resolve().parent.parent.joinpath("media/input")
OUTPUT_MEDIA = Path(__file__).resolve().parent.parent.joinpath("media/output")

# TRANSLATED_AUDIO = "{filename}_to_{target_language}{suffix}"
AUDIO_FILE = "{target_language}_{filename}"


def main() -> None:
    tta = TextToAudio()
    text = "Bark generates audio from scratch."
    path = OUTPUT_MEDIA.joinpath(
        AUDIO_FILE.format(
            filename='dog_says_hubert.wav', target_language='eng'
        ),
    )
    save_audio(
        path=path,
        audio=tta.get_audio_array(text),
        sample_rate=tta.model.generation_config.sample_rate,
    )


if __name__ == "__main__":
    main()
