# Krok 1: Importowanie potrzebnych bibliotek
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import shap # <-- WAŻNE: Dodajemy bibliotekę SHAP

# --- Konfiguracja ---
NAZWA_FOLDERU_WYNIKOW = "wyniki_iris"

# --- Główny skrypt ---

def main():
    """Główna funkcja wykonująca analizę."""
    
    # Krok 2: Tworzenie folderu na wyniki
    if not os.path.exists(NAZWA_FOLDERU_WYNIKOW):
        os.makedirs(NAZWA_FOLDERU_WYNIKOW)
        print(f"Utworzono folder: '{NAZWA_FOLDERU_WYNIKOW}'")

    # Krok 3: Wczytanie i przygotowanie danych
    iris_dataset = load_iris()
    df = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_dataset.target, iris_dataset.target_names)
    
    # ... (tutaj pomijam wydruki, które już masz, żeby nie powtarzać) ...

    # Krok 4: Eksploracyjna Analiza Danych (EDA) i Wizualizacja
    print("\nGenerowanie wykresu: Pairplot...")
    sns.pairplot(df, hue='species', palette='viridis')
    plt.suptitle("Relacje między cechami dla gatunków Irysów", y=1.02)
    plt.savefig(os.path.join(NAZWA_FOLDERU_WYNIKOW, "pairplot.png"))
    plt.close()

    print("Generowanie wykresu: Mapa korelacji...")
    numeric_df = df.drop(columns=['species'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Mapa korelacji cech Irysów")
    plt.savefig(os.path.join(NAZWA_FOLDERU_WYNIKOW, "heatmapa_korelacji.png"))
    plt.close()

    # Krok 5: Przygotowanie danych do trenowania modelu
    X = df.drop(columns=['species'])
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Krok 6: Trenowanie modelu KNN
    print("\nTrenowanie modelu K-Najbliższych Sąsiadów...")
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    print("Model został wytrenowany.")

    # Krok 7: Ocena modelu
    y_pred = model.predict(X_test)
    print("\n--- Ocena modelu na danych testowych ---")
    print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
    # ... (pomijam wydruk raportu) ...

    print("Generowanie wykresu: Macierz pomyłek...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris_dataset.target_names, 
                yticklabels=iris_dataset.target_names)
    plt.xlabel("Przewidziana etykieta")
    plt.ylabel("Prawdziwa etykieta")
    plt.title("Macierz pomyłek")
    plt.savefig(os.path.join(NAZWA_FOLDERU_WYNIKOW, "macierz_pomylek.png"))
    plt.close()

    # Krok 8: Analiza SHAP (NOWY KROK)
    print("\n--- Rozpoczynanie analizy SHAP ---")
    
    # Używamy KernelExplainer, ponieważ jest model-agnostic (działa z każdym modelem)
    # Wyjaśnia on predykcje modelu, badając, jak zmieniają się wyniki, gdy podmieniamy wartości cech
    # Jako dane tła podajemy zbiór treningowy, żeby SHAP wiedział, jakie są "typowe" wartości cech
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    
    # Obliczamy wartości SHAP dla zbioru testowego
    print("Obliczanie wartości SHAP (może to chwilę potrwać)...")
    shap_values = explainer.shap_values(X_test)
    
    # Tworzymy wykres podsumowujący SHAP
    print("Generowanie wykresu: Podsumowanie SHAP...")
    plt.figure()
    shap.summary_plot(shap_values, X_test, 
                      class_names=iris_dataset.target_names,
                      feature_names=X.columns,
                      show=False) # show=False, żeby nie wyświetlać w oknie
    plt.tight_layout() # Dopasowanie układu, żeby etykiety się nie nakładały
    sciezka_zapisu = os.path.join(NAZWA_FOLDERU_WYNIKOW, "shap_summary.png")
    plt.savefig(sciezka_zapisu)
    plt.close()
    print(f"Zapisano wykres SHAP do pliku: {sciezka_zapisu}")


if __name__ == "__main__":
    main()
