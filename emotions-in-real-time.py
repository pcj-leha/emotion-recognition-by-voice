import soundfile as sf
import numpy as np
import librosa
import pyaudio
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from PyQt5 import QtWidgets

from Ui_MainWindow import Ui_MainWindow

model = pickle.load(open("model/mlp_classifier.model", "rb"))

CHUNK_SIZE = 1024
RATE = 16000
THRESHOLD = 500
SILENCE = 30

#извлечение признаков mfcc,chroma,mel из аудиофайла
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=120).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

#Возвращает 'Истину', если ниже порогового значения 'без звука'
def is_silent(snd_data):
    return max(snd_data) < THRESHOLD

#усреднение громкости
def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r
# Обрезание пустых мест начале и конце
def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Обрезание пустых мест справа
    snd_data = _trim(snd_data)

    # Обрезание пустых мест справа
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

#Добавьте тишину в начало и конец 'and_data' длиной 'секунды'
def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Запишите слово или слова с микрофона и
    верните данные в виде массива подписанных
    коротких сообщений.

    Нормализует звук, отключает тишину с начала
    и до конца и добавляет 0,5 секунды чистого звука,
    чтобы убедиться, что все сервисы могут воспроизводить
    его без отключения.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
        input=True, output=True,
        frames_per_buffer=1024)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        snd_data = array('h', stream.read(CHUNK_SIZE)) #начали слушать

        if byteorder == 'big':
            snd_data.byteswap()

        r.extend(snd_data)

        silent = is_silent(snd_data)    # True если тишина

        if silent and snd_started:  # если начал говорить и тишина
            num_silent += 1
        elif not silent and not snd_started:    # если человек говорит и он до этого не начинал говорить
            snd_started = True

        if snd_started and num_silent > SILENCE:    # если начинал говорить и размер тишины больше заданного
            break

    sample_width = p.get_sample_size(pyaudio.paInt16)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Запись файла"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def start_emotion(ui):
    print("Please talk")
    filename = "test.wav"
    # запись файла
    record_to_file(filename)
    # извлечение признаков
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # предсказание
    result = model.predict(features)[0]
    # показ результатов
    print("result:", result)
    ui.label.setText(str(result))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.pushButton.clicked.connect(lambda: start_emotion(ui))
    sys.exit(app.exec_())
