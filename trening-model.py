#импортирование библиотек
import soundfile as sf    #читает аудиофайл
import librosa            #извлечение признаков
import os                 #работа с операционной системой
import glob               #нахождение всех путей по заданному шаблону
import pickle             #сохранение модели
import numpy as np        #представляет общие математические и числовые операции
from sklearn.model_selection import train_test_split    #разделение обучающих и тестовых данных
from sklearn.neural_network import MLPClassifier        #многослойный классификатор персептрона
from sklearn.metrics import accuracy_score              #измерение точности

#Все эмоции находящиеся в наборе данных
int_emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
#эмоции, которые мы хотим использовать
EMOTIONS = {"happy","sad","neutral","angry"}

#извлечение признаков mfcc,chroma,mel,contrast,tonnetz из аудиофайла
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

#делаем и делим результат на обучающие и тестировачные наборы данных
def train_test_data(test_size):
    features, emotions = [],[] #массивы для признаков и эмоций
    for file in glob.glob("data/Actor_*/*.wav"):
        fileName = os.path.basename(file)   #получение имени файла
        emotion = int_emotion[fileName.split("-")[2]] #получаем эмоцию
        if emotion not in EMOTIONS:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True) #извлечение признаков из аудиофайла
        features.append(feature)
        emotions.append(emotion)
    return train_test_split(np.array(features), emotions, test_size=test_size, random_state=7) #возвращаем разделённые данные для обучения и тестирования

#мы получаем данные о тренировочных и тестовых наборов из train_test_data(). Здесь размер теста составляет 25%.
X_train,X_test,y_train,y_test=train_test_data(test_size=0.25)
print("Количество наборов для обучения: ",X_train.shape[0])
print("Количество наборов для теста: ",X_test.shape[0])
print("Количество признаков: ",X_train.shape[1])

#обЪявляем многослойный классификатор персептрона
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(400,), learning_rate='adaptive', max_iter=1000)


#обучение модели
print("___________Обучение модели___________")
model.fit(X_train,y_train)

#Прогнозирование выходного значения для тестовых данных
y_pred = model.predict(X_test)

#калькулятор точности
accuracy = accuracy_score(y_true=y_test,y_pred=y_pred)
accuracy*=100
print("Точность: {:.4f}%".format(accuracy))

#сохранение модели
if not os.path.isdir("model"):
   os.mkdir("model")
pickle.dump(model, open("model/mlp_classifier.model", "wb"))
