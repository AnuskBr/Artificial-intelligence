#http://yann.lecun.com/exdb/mnist/
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
#from sklearn.metrics import ConfusionMatrixDisplay
import platform
import time
from sklearn.metrics import ConfusionMatrixDisplay

#train data
print("Date de antrenare")

if not platform.system() == 'Windows':
    X_train, y_train = loadlocal_mnist(
            images_path='train-images-idx3-ubyte', 
            labels_path='train-labels-idx1-ubyte')

else:
    X_train, y_train = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')
print('Dimensions: %s x %s' % (X_train.shape[0], X_train.shape[1]))
print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y_train))
print('Class distribution: %s' % np.bincount(y_train))
#test data
print("\nDate de test")
if not platform.system() == 'Windows':
    X_test, y_test = loadlocal_mnist(
            images_path='t10k-images-idx3-ubyte', 
            labels_path='t10k-labels-idx1-ubyte')

else:
    X_test, y_test = loadlocal_mnist(
            images_path='t10k-images.idx3-ubyte', 
            labels_path='t10k-labels.idx1-ubyte')
print('Dimensions: %s x %s' % (X_test.shape[0], X_test.shape[1]))
#print('\n1st row', X_test[0])
print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y_test))
print('Class distribution: %s' % np.bincount(y_test))

# vizualizarea primei cifre din fiecare clasa
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.savefig('images/12_5.png', dpi=300)
plt.show()


# vizualizarea a 25 de versiuni a unei cifre alese
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#clasificator
start_train = time.time()
mlp_clf = MLPClassifier(hidden_layer_sizes=(100),
                        max_iter = 300,activation = 'logistic',
                        solver = 'adam')

mlp_clf.fit(X_train, y_train)
end_train = time.time()
# masurare timp predictie
start_predict = time.time()
y_pred = mlp_clf.predict(X_test)
end_predict = time.time()

#timp pentru training si pentru predictie
train_time = end_train - start_train
predict_time = end_predict - start_predict



print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print(f"Timpul de antrenare: {train_time:.2f} seconds")
print(f"Timpul predictiei: {predict_time:.2f} seconds")
fig, ax = plt.subplots()  
ConfusionMatrixDisplay.from_estimator(mlp_clf, X_test, y_test, display_labels=mlp_clf.classes_, ax=ax)
ax.set_title("Confusion Matrix for MNIST")
plt.show()

#fig = plot_confusion_matrix(mlp_clf, X_test, y_test, display_labels=mlp_clf.classes_)
#fig.figure_.suptitle("Confusion Matrix for MNIST")
#plt.show()
print(classification_report(y_test, y_pred))