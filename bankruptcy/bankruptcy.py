# %pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

polish_companies_bankruptcy = fetch_ucirepo(id=365)
X = polish_companies_bankruptcy.data.features
y = polish_companies_bankruptcy.data.targets

### preprocessing ####
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

X = X.replace('?', np.nan)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

y_flat = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y_flat, test_size=0.2, random_state=42, stratify=y_flat )
imputer = SimpleImputer(strategy='median')
imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
scaler = StandardScaler()
scaler.fit(X_train_imputed)
X_train_scaled = scaler.transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
smote = SMOTE(random_state=42)
print(f"Rozmiar zbioru treningowego przed SMOTE: {X_train_scaled.shape}")
print(f"Rozkład klas w y_train przed SMOTE: {np.bincount(y_train)}")
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"\nRozmiar zbioru treningowego po SMOTE: {X_train_resampled.shape}")
print(f"Rozkład klas w y_train po SMOTE: {np.bincount(y_train_resampled)}")

### MLP

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall

tf.random.set_seed(42)

model_full = Sequential([Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
Dropout(0.5),Dense(32, activation='relu'),Dropout(0.5),Dense(1, activation='sigmoid') ])
model_full.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
model_full.summary()
history_full = model_full.fit(X_train_resampled, y_train_resampled,epochs=50,batch_size=256, validation_data=(X_test_scaled, y_test),verbose=1)

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

pd.DataFrame(history_full.history).plot(figsize=(10, 6))
plt.title("Historia treningu na pełnym zbiorze")
plt.grid(True)
plt.show()

results = model_full.evaluate(X_test_scaled, y_test, verbose=0)
print("\nWyniki dla pełnego zbioru")
for name, value in zip(model_full.metrics_names, results):
    print(f"{name}: {value:.4f}")

y_pred_proba = model_full.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype("int32")
print("\nRaport klasyfikacji dla pełnego zbioru:")
print(classification_report(y_test, y_pred, target_names=['Działająca', 'Bankrut']))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Działająca', 'Bankrut'], yticklabels=['Działająca', 'Bankrut'])
plt.xlabel('Przewidziane')
plt.ylabel('Prawdziwe')
plt.title('Macierz Pomyłek dla pełnego zbioru')
plt.show()

### shap

import shap


background_data_resampled = shap.sample(X_train_resampled, 100)
explainer = shap.KernelExplainer(model_full.predict, background_data_resampled)
X_test_sample = shap.sample(X_test_scaled, 200) 
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample, feature_names=X.columns)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

shap_values_flat = np.abs(shap_values[1]).flatten()
print(f"Liczba cech (kolumn): {len(X.columns)}")
print(f"Liczba obliczonych ważności: {len(shap_values_flat)}")
feature_importance_df = pd.DataFrame({
    'Cecha': X.columns,
    'Ważność SHAP': shap_values_flat
})

feature_importance_df = feature_importance_df.sort_values(by='Ważność SHAP', ascending=False)
plt.figure(figsize=(10, 8))
plt.title('Najważniejsze cechy wg sredniej wartosci Shap')
plt.barh(
    feature_importance_df['Cecha'][:15][::-1], 
    feature_importance_df['Ważność SHAP'][:15][::-1]
)
plt.xlabel("Średnia absolutna wartość SHAP (wpływ na predykcję modelu)")
plt.grid(axis='x')
plt.show()

### redukcja cech i ponowny trening

num_features_to_drop = 15
features_to_drop = feature_importance_df.tail(num_features_to_drop)['Cecha'].tolist()
print(f"\nCechy do usunięcia: {features_to_drop}")
X_train_reduced = X_train.drop(columns=features_to_drop)
X_test_reduced = X_test.drop(columns=features_to_drop)
imputer_reduced = SimpleImputer(strategy='median')
X_train_reduced_imputed = imputer_reduced.fit_transform(X_train_reduced)
X_test_reduced_imputed = imputer_reduced.transform(X_test_reduced)
scaler_reduced = StandardScaler()
X_train_reduced_scaled = scaler_reduced.fit_transform(X_train_reduced_imputed)
X_test_reduced_scaled = scaler_reduced.transform(X_test_reduced_imputed)
smote_reduced = SMOTE(random_state=42)
X_train_reduced_resampled, y_train_reduced_resampled = smote_reduced.fit_resample(X_train_reduced_scaled, y_train)
model_reduced = Sequential([ Dense(64, activation='relu', input_shape=(X_train_reduced_resampled.shape[1],)),
Dropout(0.5),Dense(32, activation='relu'),Dropout(0.5),Dense(1, activation='sigmoid')])

model_reduced.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

model_reduced.summary()
history_reduced = model_reduced.fit(X_train_reduced_resampled, y_train_reduced_resampled,
epochs=50,batch_size=256,validation_data=(X_test_reduced_scaled, y_test),verbose=0 )


### porownanie obu modeli 

print("\nModel z pełnym zbiorem cech:")
results_full = model_full.evaluate(X_test_scaled, y_test, verbose=0)
for name, value in zip(model_full.metrics_names, results_full):
    print(f"{name}: {value:.4f}")

print("\nModel ze zmniejszonym zbiorem cech:")
results_reduced = model_reduced.evaluate(X_test_reduced_scaled, y_test, verbose=0)
for name, value in zip(model_reduced.metrics_names, results_reduced):
    print(f"{name}: {value:.4f}")
