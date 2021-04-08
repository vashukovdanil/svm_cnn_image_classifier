from keras.preprocessing import image
from skimage.color import rgb2gray
from skimage.feature import hog
import tensorflow as tf
import numpy as np
import svm
import os

# Flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Загрузка приложения
app = Flask(__name__, template_folder='templates')

# Загрузка моделей
cnn = tf.keras.models.load_model('cnn_model.h5')
svm = svm.train_model()


# Получить предсказание для CNN (нейронные сети)
def cnn_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Подготовка изображения
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])

    pred = model.predict(images)
    return pred


# Получить предсказание для метода опорных векторов (машинное обучение)
def svm_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    gray_img = rgb2gray(img)
    X = hog(gray_img, block_norm='L2-Hys', pixels_per_cell=(16, 16)).reshape(-1, 1).T
    pred = model.predict(X)
    return pred


def format_results(svm, cnn):
    svm = svm[0]
    if round(cnn[0][0]) == 1:
        cnn = "Брюшная полость"
    else:
        cnn = "Грудная клетка"
    
    if svm == "abd":
        svm = "Брюшная полость"
    else:
        svm = "Грудная клетка"
    result = f'Метод опорных векторов (SVM) - {svm}, Свёрточная нейронная сеть (CNN) - {cnn}'
    return result

# Главная страница
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Вывод результатов
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Получить файл из запроса POST
        f = request.files['file']

        # Сохранить этот файл в папку /uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Классифицировать
        cnn_preds = cnn_predict(file_path, cnn)
        svm_preds = svm_predict(file_path, svm)

        # Вывод результатов
        result = format_results(svm_preds, cnn_preds)
        return result
    return None


if __name__ == "__main__":
    app.run()