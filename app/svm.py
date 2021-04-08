import os

# Графика
import matplotlib as mpl
import matplotlib.pyplot as plt

# Работа с данными
import pandas as pd
import numpy as np

# Работа с растровой графикой
from PIL import Image, ImageDraw

# Обработка изображений
from skimage.feature import hog
from skimage.color import rgb2gray

# Обучение
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC





def get_image(id, root_dir):
    # Открывает изображение в папке и возвращает его в качестве массива
    file = f'{id}.png'
    path = os.path.join(root_dir, file)
    img = Image.open(path)
    img = img.resize((224,224), Image.ANTIALIAS)
    img_arr = np.array(img)

    return img_arr

def create_features(img):
    # Делает изображение серым и сглаживает его до одной строки
    gray_img = rgb2gray(img)
    flat_features = hog(gray_img, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    return flat_features


def create_feature_matix(root_path):
    # Пробегает по всем папкам-классам и
    # создаёт матрицу по всех изображениям
    
    names = ["abd", "chst"]
    matrix = []
    labels = []

    for set_name in names:
        root = f"{root_path}/{set_name}/"
        f_counts = len([name for name in os.listdir(root)])

        for i in range(1, f_counts+1):
            img = get_image(i, root)
            img_ft = create_features(img)
            matrix.append(img_ft)
            labels.append(set_name)
    
    matrix = pd.DataFrame(matrix)
    return labels, matrix



def train_model():
    # Тренировка модели. Возвращает обученную модель.
    y_train, X_train = create_feature_matix('train')
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    return svm