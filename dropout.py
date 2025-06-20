# %pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np

#wykorztstujemy zbior dotyczacy sukcesu akademickiego studentow
df = pd.read_csv('dropout.csv', sep=';')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

#### preprocessing ###########

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Target'])
class_names = label_encoder.classes_
y_one_hot = to_categorical(y_encoded)
X = df.drop('Target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

####budowa i trening modelu MLP

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

tf.random.set_seed(42)


model_full = Sequential([Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)), Dropout(0.3), Dense(64, activation='relu'), Dropout(0.3),Dense(y_train.shape[1], activation='softmax') ])
model_full.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model_full.summary()
history_full = model_full.fit(X_train_scaled, y_train,epochs=50,batch_size=32,validation_data=(X_test_scaled, y_test),verbose=1 )

### sprawdzamy jak model radzi sobie na danych testowych

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

pd.DataFrame(history_full.history).plot(figsize=(8, 5))
plt.title("Historia treningu modelu")
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
loss, accuracy = model_full.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nDokładność na zbiorze testowym: {accuracy*100:.2f}%")
y_pred_proba = model_full.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nRaport klasyfikacji :")
print(classification_report(y_true, y_pred, target_names=class_names))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Przewidziane')
plt.ylabel('Prawdziwe')
plt.title('Macierz Pomyłek')
plt.show()

### wykorzystanie shapa do zobaczenia co jest w srodku modelu

import shap

explainer = shap.KernelExplainer(model_full.predict, background_data)
shap_values = explainer.shap_values(X_test_scaled)
print("\nWykres SHAP:")
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, class_names=class_names)

## redukowanie cech i kolejny trening


mean_abs_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
feature_importance = pd.DataFrame(list(zip(X.columns, mean_abs_shap)), columns=['Cecha', 'Średnia wart. SHAP'])
feature_importance = feature_importance.sort_values(by='Średnia wart. SHAP', ascending=True)
num_features_to_drop = 5
features_to_drop = feature_importance.head(num_features_to_drop)['Cecha'].tolist()
print(f"\nwybrane najmniej istotne cechy do usunicia: {features_to_drop}")
X_train_reduced = X_train.drop(columns=features_to_drop)
X_test_reduced = X_test.drop(columns=features_to_drop)
scaler_reduced = StandardScaler()
X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced)
model_reduced = Sequential([Dense(128, activation='relu', input_shape=(X_train_reduced_scaled.shape[1],)),Dropout(0.3),Dense(64, activation='relu'),Dropout(0.3),Dense(y_train.shape[1], activation='softmax')])
model_reduced.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history_reduced = model_reduced.fit(X_train_reduced_scaled, y_train,epochs=50,batch_size=32,validation_data=(X_test_reduced_scaled, y_test),verbose=0)

###porownanie wynikow

loss_reduced, accuracy_reduced = model_reduced.evaluate(X_test_reduced_scaled, y_test, verbose=0)
print(f"\nDokładność na zbiorze testowym (zredukowany zbiór): {accuracy_reduced*100:.2f}%")
print(f"Model z pełnym zbiorem cech: Dokładność = {accuracy*100:.2f}%")
print(f"Model ze zrmniejszonym zbiorem cech: Dokładność = {accuracy_reduced*100:.2f}%")
y_pred_reduced = np.argmax(model_reduced.predict(X_test_reduced_scaled), axis=1)
print("\nRaport klasyfikacji (zredukowany zbiór):")
print(classification_report(y_true, y_pred_reduced, target_names=class_names))
