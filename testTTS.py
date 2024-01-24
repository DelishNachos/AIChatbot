import torchaudio
from speechbrain.pretrained import FastSpeech2
from speechbrain.pretrained import HIFIGAN

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
fastspeech2 = FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

# Run TTS with text input
input_text = "were the leaders in this luckless change; though our own Baskerville; who was at work some years before them; went much on the same lines;"

mel_output, durations, pitch, energy = fastspeech2.encode_text(
  [input_text],
  pace=1.0,        # scale up/down the speed
  pitch_rate=1.0,  # scale up/down the pitch
  energy_rate=1.0, # scale up/down the energy
)

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
torchaudio.save('example_TTS_input_text.wav', waveforms.squeeze(1), 22050)