import pyaudio
import wave
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tensorflow import keras
import numpy as np
from matplotlib import image

model = keras.models.load_model('/home/james/Downloads/CNNModel.h5')
# wheezes = load('/home/james/Downloads/wheezedata.npy', allow_pickle=True)
# both = load('/home/james/Downloads/bothdata.npy', allow_pickle=True)
# crackles = load('/home/james/Downloads/cracklesdata.npy', allow_pickle=True)
# none = load('/home/james/Downloads/data.npy', allow_pickle=True)
clip = []
# clip.append(none[1][0].reshape(50, 245, 1))
# clip.append(wheezes[1][0].reshape(50, 245, 1))
# clip.append(wheezes[2][0].reshape(50, 245, 1))
# clip.append(wheezes[2][0].reshape(50, 245, 1))
# clip.append(both[8][0].reshape(50, 245, 1))
# clip.append(both[4][0].reshape(50, 245, 1))
# clip.append(none[5][0].reshape(50, 245, 1))
# clip.append(crackles[8][0].reshape(50, 245, 1))
# clip.append(crackles[4][0].reshape(50, 245, 1))
# clip.append(crackles[5][0].reshape(50, 245, 1))


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22000
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "resources/test.wav"


def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig('sp_xyz.png', dpi=300)


def record():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

    return wave_file


audio = record()
audio_fpath = "resources/"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ", len(audio_clips))
graph_spectrogram('resources/test.wav')

image = image.imread('sp_xyz.png')

audio_clip = image.resize(50, 245, 1)
audio_clip = np.array(image, dtype=np.float32)
clip.append(audio_clip.reshape(1, 50, 245, 1))

prediction = model.predict(clip)
classes = np.argmax(prediction, axis=1)
print(classes)
