import pyaudio
import wave
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import librosa
import librosa.display
import soundfile as sf

model = keras.models.load_model('/home/james/Downloads/Jan21RespModel.h5')

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22000
CHUNK = 512
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "resources/test.wav"


def audio_features(filename):
    sound, sample_rate = sf.read(filename)
    stft = np.abs(librosa.stft(sound))
    mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=8000, n_mfcc=40, fmin=30).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=8000).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(sound, sr=8000, fmin=30).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=8000, fmin=30).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate,
                                              chroma=librosa.feature.chroma_cqt(y=sound, sr=8000, fmin=30)).T, axis=0)
    concat = np.concatenate((mfccs, chroma, mel, contrast, tonnetz))
    return concat


def graph_spectrogram(S):
    plt.subplot(2, 2, 1)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.tight_layout()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    plt.savefig('sp_xyz.png', bbox_inches=None, pad_inches=0, dpi=300)


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
x = audio_features('resources/test.wav')
print(x)
S = librosa.feature.melspectrogram(x)
graph_spectrogram(S)
x = np.array(x)
x = np.reshape(x, [1, 193, 1])

prediction = model.predict(x)
classes = np.argmax(prediction, axis=1)
print(classes)
