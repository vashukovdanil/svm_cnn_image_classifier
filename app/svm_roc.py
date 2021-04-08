import svm as model
from sklearn.metrics import roc_curve, auc

# Графика
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

svm_model = model.train_model()

y_test, X_test = model.create_feature_matix('test')


probabilities = svm_model.predict_proba(X_test)
y_proba = probabilities[:, 1]


y_new = []
for el in y_test:
    if el == 'abd':
        y_new.append(0)
    else:
        y_new.append(1)
y_test = y_new
print(y_proba)



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)
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