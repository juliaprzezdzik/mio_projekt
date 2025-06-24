# Krok 1: Importowanie potrzebnych bibliotek
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # <-- WAŻNE: Będziemy skalować dane
from sklearn.linear_model import LogisticRegression # <-- Zmieniamy model dla różnorodności
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import shap

# --- Konfiguracja ---
NAZWA_FOLDERU_WYNIKOW = "wyniki_wine"

# --- Główny skrypt ---

def main():
    """Główna funkcja wykonująca analizę."""
    
    # Krok 2: Tworzenie folderu na wyniki
    if not os.path.exists(NAZWA_FOLDERU_WYNIKOW):
        os.makedirs(NAZWA_FOLDERU_WYNIKOW)
        print(f"Utworzono folder: '{NAZWA_FOLDERU_WYNIKOW}'")

    # Krok 3: Wczytanie i przygotowanie danych
    wine_dataset = load_wine()
    df = pd.DataFrame(data=wine_dataset.data, columns=wine_dataset.feature_names)
    df['target'] = wine_dataset.target
    df['target_name'] = df['target'].map({i: name for i, name in enumerate(wine_dataset.target_names)})

    print("--- Pierwsze 5 wierszy danych ---")
    print(df.head())
    print("\n--- Statystyki opisowe (zwróć uwagę na różne skale cech!) ---")
    print(df.describe())

    # Krok 4: Preprocessing i przygotowanie danych do modelu
    # X -> cechy (dane wejściowe)
    # y -> etykiety (to, co chcemy przewidzieć)
    X = df.drop(columns=['target', 'target_name'])
    y = df['target']
    
    # Dzielimy dane na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # !!! WAŻNY KROK: Skalowanie cech !!!
    # Cechy mają różne zakresy (np. 'alcohol' vs 'proline'). Skalowanie (standaryzacja)
    # sprowadza je do podobnego zakresu, co jest ważne dla wielu modeli, w tym regresji logistycznej.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Konwertujemy z powrotem do DataFrame, żeby zachować nazwy kolumn dla wykresów SHAP
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"\nRozmiar zbioru treningowego: {X_train.shape[0]} próbek")
    print(f"Rozmiar zbioru testowego: {X_test.shape[0]} próbek")
    print("Dane zostały przeskalowane za pomocą StandardScaler.")

    # Krok 5: Trenowanie modelu - Regresja Logistyczna
    # Zmieniamy model, aby pokazać analizę dla innego algorytmu.
    # Regresja logistyczna to silny, standardowy model klasyfikacyjny.
    print("\nTrenowanie modelu Regresji Logistycznej...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model został wytrenowany.")

    # Krok 6: Ocena modelu
    print("\n--- Ocena modelu na danych testowych ---")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność (Accuracy): {accuracy:.4f}")
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=wine_dataset.target_names))

    # Wykres 1: Macierz pomyłek
    print("Generowanie wykresu: Macierz pomyłek...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=wine_dataset.target_names, 
                yticklabels=wine_dataset.target_names)
    plt.xlabel("Przewidziana etykieta")
    plt.ylabel("Prawdziwa etykieta")
    plt.title("Macierz pomyłek - Zbiór Wine")
    sciezka_zapisu = os.path.join(NAZWA_FOLDERU_WYNIKOW, "macierz_pomylek_wine.png")
    plt.savefig(sciezka_zapisu)
    plt.close()
    print(f"Zapisano wykres do pliku: {sciezka_zapisu}")
    
    # Wykres 2: Heatmapa korelacji
    print("\nGenerowanie wykresu: Mapa korelacji...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".1f", annot_kws={"size": 8})
    plt.title("Mapa korelacji cech w zbiorze Wine")
    plt.tight_layout()
    sciezka_zapisu = os.path.join(NAZWA_FOLDERU_WYNIKOW, "heatmapa_korelacji_wine.png")
    plt.savefig(sciezka_zapisu)
    plt.close()
    print(f"Zapisano wykres do pliku: {sciezka_zapisu}")


    # Krok 7: Analiza SHAP
    print("\n--- Rozpoczynanie analizy SHAP ---")
    
    # Jako funkcję predykcji podajemy model.predict_proba
    explainer = shap.KernelExplainer(model.predict_proba, X_train_scaled)
    
    # Obliczamy wartości SHAP dla przeskalowanego zbioru testowego
    print("Obliczanie wartości SHAP (może to chwilę potrwać)...")
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Tworzymy wykres podsumowujący SHAP
    print("Generowanie wykresu: Podsumowanie SHAP...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_scaled_df, 
                      class_names=wine_dataset.target_names,
                      show=False)
    plt.title("Wpływ cech na predykcję modelu (SHAP) - Zbiór Wine")
    plt.tight_layout()
    sciezka_zapisu = os.path.join(NAZWA_FOLDERU_WYNIKOW, "shap_summary_wine.png")
    plt.savefig(sciezka_zapisu)
    plt.close()
    print(f"Zapisano wykres SHAP do pliku: {sciezka_zapisu}")


if __name__ == "__main__":
    main()
