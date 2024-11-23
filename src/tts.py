from transformers import BarkModel, AutoProcessor
from torch import Tensor

DEFAULT_VOICE_PRESET = None  # "v2/en_speaker_6"
TTS_MODEL_NAME = "suno/bark"


class TextToAudio:

    def __init__(
        self,
        model_name: str = TTS_MODEL_NAME,
        voice_preset: str | None = DEFAULT_VOICE_PRESET,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BarkModel.from_pretrained(model_name)
        self.voice_preset = voice_preset

    def get_audio_array(
        self,
        text: str,
    ) -> Tensor:
        return self.model.generate(
            **self.processor(
                text,
                voice_preset=self.voice_preset,
            ),
        )[0].cpu()
