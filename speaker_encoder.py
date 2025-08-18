
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import logging

# Set up logging
logger = logging.getLogger(__name__)

class SpeakerEncoder:
    """
    A class to handle loading a speaker encoder model and extracting embeddings.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the speaker encoder model.
        """
        self.device = device
        logger.info(f"Initializing SpeakerEncoder on device: {self.device}")
        try:
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            logger.info("Speaker encoder model (ECAPA-TDNN) loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load speaker encoder model: {e}")
            raise

    @torch.no_grad()
    def get_embedding(self, audio_path: str) -> torch.Tensor:
        """
        Generates a speaker embedding from a given audio file path.

        Args:
            audio_path: The path to the input audio file.

        Returns:
            A torch.Tensor containing the speaker embedding.
        """
        try:
            signal, fs = torchaudio.load(audio_path)

            # Ensure signal is on the correct device
            signal = signal.to(self.device)

            # The model expects 16kHz audio. Resample if necessary.
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000).to(self.device)
                signal = resampler(signal)

            # The model works best with mono audio
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)

            # Get the embedding
            embedding = self.model.encode_batch(signal)
            
            # Reshape to [1, 1, 192] as expected by the TTS model's embedding layer
            embedding = embedding.squeeze().unsqueeze(0).unsqueeze(0)
            
            logger.info(f"Successfully generated speaker embedding from {audio_path}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for {audio_path}: {e}")
            # Return a zero tensor or raise an exception if you prefer
            return torch.zeros((1, 1, 192), device=self.device)
