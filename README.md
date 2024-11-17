
# Fraud Detection Using Artificial Neural Network

![Project Image](https://miro.medium.com/max/1200/0*_6WEDnZubsQfTMlY.png)

> Detecting customers who may commit fraud

---

### Table of Contents

- [Description](#description)
- [Data Preprocessing](#data-preprocessing)
- [Artificial Neural Network](#artificial-neural-network)
- [Statistical Evaluation](#statistical-evaluation)
- [Undersampling Technique](#undersampling-technique)
- [Statistical Evaluation of Undersampling Technique](#statistical-evaluation-of-undersampling-technique)
- [Comparison of Initial and Undersampled Models](#comparison-of-initial-and-undersampled-models)
- [About the Author](#about-the-author)

---

## Description

This project aims to help a credit card company detect customers potentially involved in fraud. The company provided a [CSV file](https://biy.ly39g52lF) containing data for 280,000 users with 29 independent variables and one dependent variable. The goal is to create a model that classifies whether a customer is likely to commit fraud or not.

#### Technologies

- Python
- Artificial Neural Network
- Undersampling Technique

[Back To The Top](#fraud-detection-using-artificial-neural-network)

---

## Data Preprocessing

#### Feature Engineering & Selection
[**Feature Engineering**](http://belajardatascience.blogspot.com/2018/05/feature-engineering.html) refers to creating or modifying features using domain knowledge, while **Feature Selection** involves selecting or combining features to enhance the accuracy of the machine learning model.

#### Importing Libraries and Data
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('credit_card_data.csv')
data.shape
```

#### Feature Engineering
Standardizing the *Amount* column using "StandardScaler"
```python
from sklearn.preprocessing import StandardScaler
data['standardized_amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
```
This standardization simplifies computations.

#### Feature Selection
Removing unused variables.
```python
y = np.array(data.iloc[:, -2])
X = np.array(data.drop(['Time', 'Amount', 'Class'], axis=1))
```
- `y`: Dependent variable  
- `X`: Independent variables  

---

## Artificial Neural Network
To train the model to detect fraudulent users, we used an [Artificial Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network). This method is an adaptive system that changes its structure to solve problems based on internal and external information flowing through the network.

#### Splitting Training and Test Sets
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=111)
```

#### Building the Model
```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Dense(units=16, input_dim=29, activation='relu'))
classifier.add(Dense(units=24, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units=20, activation='relu'))
classifier.add(Dense(units=24, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()
```

#### Visualizing the Model
```python
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model_ann.png', show_shapes=True, show_layer_names=False)
```
![model ann](https://raw.githubusercontent.com/albarabimakasa/solve-credit-card-problem-with-ANN-model/main/picture/model%20ann.png)

#### Training the ANN Model
```python
run_model = classifier.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1, validation_data=(X_validate, y_validate))
```

---

## Statistical Evaluation

#### Accuracy Visualization
```python
plt.plot(run_model.history['accuracy'])
plt.plot(run_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.show()
```
![accuracy](https://raw.githubusercontent.com/albarabimakasa/solve-credit-card-problem-with-ANN-model/main/picture/akurasi.png)

#### Loss Visualization
```python
plt.plot(run_model.history['loss'])
plt.plot(run_model.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.show()
```
![loss](https://raw.githubusercontent.com/albarabimakasa/solve-credit-card-problem-with-ANN-model/main/picture/loss.png)

#### Testing the Model on Unseen Data
```python
evaluation = classifier.evaluate(X_test, y_test)
print('Accuracy: {:.2f}%'.format(evaluation[1] * 100))
```
> Accuracy: 99.94%

---

## Undersampling Technique
[**Undersampling**](https://socs.binus.ac.id/2019/12/26/imbalanced-dataset/) balances the dataset by reducing the size of the majority class. This approach ensures that the dataset is suitable for further modeling.  

#### Feature Engineering and Splitting
```python
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = np.array(data[data.Class == 0].index)
selected_normal_indices = np.random.choice(normal_indices, len(fraud_indices), replace=False)

final_indices = np.concatenate([fraud_indices, selected_normal_indices])
balanced_data = data.iloc[final_indices, :]

y_balanced = np.array(balanced_data.iloc[:, -2])
X_balanced = np.array(balanced_data.drop(['Time', 'Amount', 'Class'], axis=1))
```

#### Model and Training with Undersampled Data
The same steps as the initial model were repeated, but this time with the balanced dataset.



## About the Author

![Author Image](https://raw.githubusercontent.com/albarabimakasa/albarabimakasa/main/merbabu.jpeg)

Hi, I'm Albara, an industrial engineering student from the Islamic University of Indonesia with a passion for data science. Feel free to connect with me:

- Twitter - [@albara_bimakasa](https://twitter.com/albara_bimakasa)  
- Email - [18522360@students.uii.ac.id](mailto:18522360@students.uii.ac.id)

[Back To The Top](#fraud-detection-using-artificial-neural-network)
