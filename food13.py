'''
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Load the pre-trained MusicGen model
MODEL = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(MODEL)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the music prompt
prompt = (
    "A slow, soulful romantic melody featuring gentle acoustic guitar strumming, expressive sitar phrases, "
    "and airy flute melodies. The music has a soft, dreamy atmosphere with a warm, emotional tone, reminiscent "
    "of a heartfelt Bollywood love song. Gentle percussions subtly support the rhythm, allowing space for melodic "
    "improvisation. The overall mood is nostalgic and tender, evoking feelings of longing, intimacy, and peaceful love."
)

# Process the text input
inputs = processor(text=[prompt], return_tensors='pt').to(device)

# Generation configuration
model.generation_config.do_sample = True
model.generation_config.guidance_scale = 3.0
model.generation_config.max_new_tokens = 50 * 18  # ~18 seconds

# Generate audio
audio = model.generate(**inputs)

# Save output audio
sr = model.config.audio_encoder.sampling_rate  # Usually 32_000 Hz
torchaudio.save("text_music.wav", audio[0].cpu(), sr)

print("✅ Saved: text_music.wav")
'''