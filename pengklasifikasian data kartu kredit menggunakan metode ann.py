# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 20:57:37 2021

@author: USER
"""

import pydot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_ku = pd.read_csv('data_kartu_kredit.csv');
data_ku.shape

from sklearn.preprocessing import StandardScaler
data_ku['standar'] = StandardScaler().fit_transform(data_ku['Amount'].values.reshape(-1,1))

y = np.array(data_ku.iloc[:,-2])
X = np.array(data_ku.drop(['Time','Amount','Class'], axis=1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 ,
                                                            random_state=111)

X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train,
                                                            test_size=0.2,
                                                            random_state=111)

from keras.models import Sequential
from keras.layers import Dense, Dropout
classifier = Sequential()
classifier.add(Dense(units=16, input_dim=29, activation='relu'))
classifier.add(Dense(units=24, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units=20, activation='relu'))
classifier.add(Dense(units=24, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=('accuracy'))
classifier.summary()

from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file ='model_ann.png', show_shapes=True,
           show_layer_names=False)

run_model = classifier.fit(X_train, y_train,
                           batch_size = 32,
                           epochs= 5,
                           verbose = 1,
                           validation_data = (X_validate, y_validate))

print(run_model.history.keys())

plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('accuracy')
plt.ylabel('epoch')
plt.legend(['train','validate'], loc ='upper left')
plt.show()

plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.title('model loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['train','validate'], loc ='upper left')
plt.show()

evaluasi = classifier.evaluate(X_test,y_test)
print('akurasi:{:.2f}'.format(evaluasi[1]*100))

hasil_prediksi = classifier.predict_classes(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, hasil_prediksi)
cm_label = pd.DataFrame(cm, columns=np.unique(y_test),index=np.unique(y_test))
cm_label.index.name = 'aktual'
cm_label.columns.name = 'prediksi'

sns.heatmap(cm_label, annot=True, Cmap='Reds', fmt='g')


from sklearn.metrics import classification_report
jumlah_kategori = 2
target_names = ['class{}'.format(i) for i in range(jumlah_kategori)]
print(classification_report(y_test, hasil_prediksi,target_names=target_names))





#under sampling method



index_fraud = np.array(data_ku[data_ku.Class == 1].index)
n_fraud = len(index_fraud)
index_normal = np.array(data_ku[data_ku == 0].index)
index_data_normal = np.random.choice(index_normal, n_fraud, replace=False )
index_data_baru = np.concatenate([index_fraud, index_data_normal])
data_baru = data_ku.iloc[index_data_baru,:]

y_baru = np.array(data_baru.iloc[:,-2])
X_baru = np.array(data_baru.drop(['Time','Amount','Class'], axis=1))


X_train2, X_test_final, y_train2, y_test_final = train_test_split(X_baru,y_baru,
                                                                  test_size = 0.1,
                                                                  random_state=111)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train2,y_train2,
                                                        test_size = 0.1 ,
                                                        random_state=111)


X_train2, X_validate2, y_train2, y_validate2 = train_test_split(X_train2,y_train2,
                                                            test_size=0.2,
                                                            random_state=111)


classifier2 = Sequential()
classifier2.add(Dense(units=16, input_dim=29, activation='relu'))
classifier2.add(Dense(units=24, activation='relu'))
classifier2.add(Dropout(0.25))
classifier2.add(Dense(units=20, activation='relu'))
classifier2.add(Dense(units=24, activation='relu'))
classifier2.add(Dense(units=1, activation='sigmoid'))
classifier2.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=('accuracy'))
classifier.summary()


run_model2 = classifier2.fit(X_train2, y_train2,
                           batch_size = 8,
                           epochs= 5,
                           verbose = 1,
                           validation_data = (X_validate2, y_validate2))

print(run_model2.history.keys())

plt.plot(run_model2.history['accuracy'])
plt.plot(run_model2.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('accuracy')
plt.ylabel('epoch')
plt.legend(['train','validate'], loc ='upper left')
plt.show()

plt.plot(run_model2.history['loss'])
plt.plot(run_model2.history['val_loss'])
plt.title('model loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['train','validate'], loc ='upper left')
plt.show()

evaluasi2 = classifier2.evaluate(X_test2,y_test2)
print('akurasi:{:.2f}'.format(evaluasi2[1]*100))

hasil_prediksi2 = classifier2.predict_classes(X_test2)

cm2 = confusion_matrix(y_test2, hasil_prediksi2)
cm_label2 = pd.DataFrame(cm2, columns=np.unique(y_test2),index=np.unique(y_test2))
cm_label2.index.name = 'aktual'
cm_label2.columns.name = 'prediksi'

sns.heatmap(cm_label2, annot=True, Cmap='Reds', fmt='g')


print(classification_report(y_test2, hasil_prediksi2,target_names=target_names))

#pengujian model standart dan under sampling

hasil_prediksi3 = classifier2.predict_classes(X_test_final)
cm3 = confusion_matrix(y_test_final, hasil_prediksi3)
cm_label3 = pd.DataFrame(cm3, columns=np.unique(y_test_final),index=np.unique(y_test_final))
cm_label3.index.name = 'aktual'
cm_label3.columns.name = 'prediksi'
sns.heatmap(cm_label3, annot=True, Cmap='Reds', fmt='g')

print(classification_report(y_test_final, hasil_prediksi3,target_names=target_names))

#save model
classifier2.save('model_ann_kartu_kredit.hd5',include_optimizer=True)
print('alhamdulillah model sudah disimpan :) have an nice day')

