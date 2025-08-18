# Voice Cloning Process

This document outlines the step-by-step process of how voice cloning is implemented in this project.

## Step 1: Speaker Embedding Extraction

The first step in voice cloning is to extract a speaker embedding from a reference audio file. This embedding is a numerical representation of the speaker's voice characteristics.

- **Model:** The `speechbrain/spkrec-ecapa-voxceleb` model is used as the speaker encoder. This is a pre-trained model that is specifically designed for speaker recognition tasks.

- **Process:**
    1. The reference audio file is loaded.
    2. The audio is resampled to 16kHz and converted to mono.
    3. The pre-processed audio is then fed into the speaker encoder model, which generates a 192-dimensional speaker embedding.

- **Implementation:** This process is implemented in the `SpeakerEncoder` class in the `speaker_encoder.py` file. The `get_embedding` method takes an audio file path as input and returns the speaker embedding as a PyTorch tensor.

## Step 2: Text-to-Speech Synthesis with Speaker Embedding

Once the speaker embedding is extracted, it is used to condition the Text-to-Speech (TTS) model. This allows the TTS model to generate speech in the voice of the reference speaker.

- **Process:**
    1. The speaker embedding is passed as a parameter to the TTS model during inference.
    2. The TTS model uses this embedding to adapt its output, generating speech that matches the voice characteristics of the reference speaker.

- **Implementation:**
    - The `AudioDecoder` class in `flow_inference.py` takes the speaker embedding as an input to its `token2wav` method.
    - The `model_worker.py` file orchestrates the process. The `generate_stream` method in the `ModelWorker` class takes the reference audio, generates the speaker embedding using the `SpeakerEncoder`, and then passes it to the TTS model.

## Step 3: API Endpoint

The voice cloning functionality is exposed through the `/worker_generate_stream` API endpoint.

- **Endpoint:** `/worker_generate_stream`
- **Parameter:** `voice_clone_audio`
- **Usage:** To clone a voice, you need to send a POST request to the endpoint with a JSON payload that includes the `voice_clone_audio` parameter. The value of this parameter should be a base64-encoded audio file.

**Example Request:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hello, this is a test of voice cloning."
    }
  ],
  "voice_clone_audio": "base64_encoded_audio_string"
}
```
