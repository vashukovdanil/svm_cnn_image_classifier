import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math


cnn = tf.keras.models.load_model('cnn_model.h5')


X_test = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
X_test = X_test.flow_from_directory(directory='test/', target_size=(224,224), batch_size=10)

y_test = []

number_of_examples = len(X_test.filenames)
number_of_generator_calls = math.ceil(number_of_examples / (1.0 * 10))

for i in range(0,int(number_of_generator_calls)):
    y_test.extend(np.array(X_test[i][1]))


probabilities = cnn.predict(X_test)

y_proba = probabilities[:, 1]
y_test = np.array(y_test)[:, 1]


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('ROC')
roc_plot = plt.plot(false_positive_rate,
                        true_positive_rate,
                        label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()