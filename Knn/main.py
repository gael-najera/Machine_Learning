import numpy as np
import pandas as pd
from collections import Counter


class ClasificadorKNN:
    def __init__(self, k=3):
        # Inicializamos con el número de vecinos (k)
        self.k = k
        self.X_train = None
        self.y_train = None

    def entrenar(self, X, y):
        # Guardar los datos en memoria
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def clasificar(self, elemento):
        elemento = np.array(elemento)
        distancias = []

        # 1. Calculamos la distancia euclidiana contra TODOS los datos de entrenamiento
        for i in range(len(self.X_train)):
            dist = np.linalg.norm(elemento - self.X_train[i])
            distancias.append((dist, self.y_train[i]))

        # 2. Ordenamos de menor a mayor distancia
        distancias.sort(key=lambda x: x[0])

        # 3. Nos quedamos solo con los 'k' vecinos más cercanos
        k_vecinos = distancias[:self.k]

        # 4. Extraemos las etiquetas (clases) de esos vecinos
        etiquetas_vecinos = [vecino[1] for vecino in k_vecinos]

        # 5. Votación por mayoría
        votos = Counter(etiquetas_vecinos)
        clase_predicha = votos.most_common(1)[0][0]

        return clase_predicha


if __name__ == "__main__":
    archivo = 'iris.csv'

    try:
        # 1. Cargar datos
        df = pd.read_csv(archivo)
        print(f"Archivo '{archivo}' cargado. Dimensiones totales: {df.shape}")

        # 2. DIVISIÓN DATOS (80% Entrenamiento, 20% Prueba)
        columna_clase = df.columns[-1]

        # Obtenemos los índices (filas) del 80% de los datos por cada clase
        muestras_train = df.groupby(columna_clase, group_keys=False).apply(
            lambda x: x.sample(frac=0.8, random_state=42))
        indices_train = muestras_train.index

        # Filtramos el dataframe original para mantener las dimensiones intactas
        df_train = df.loc[indices_train]
        df_test = df.drop(indices_train)

        # 3. Separar X (características) e y (etiquetas)
        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values

        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values

        print(f"Datos de entrenamiento: {len(X_train)} muestras")
        print(f"Datos de prueba: {len(X_test)} muestras")

        # 4. Entrenar el modelo
        k_elegido = 3
        clf = ClasificadorKNN(k=k_elegido)
        clf.entrenar(X_train, y_train)
        print(f"\n--- Modelo entrenado con K={k_elegido} ---")

        # 5. PRUEBA DE EFECTIVIDAD
        print("\n--- Iniciando prueba de clasificación ---")
        aciertos = 0
        total_muestras_prueba = len(X_test)

        for i in range(total_muestras_prueba):
            elemento_a_probar = X_test[i]
            clase_real = y_test[i]

            # El algoritmo predice
            prediccion = clf.clasificar(elemento_a_probar)

            # Comparamos resultado
            if prediccion == clase_real:
                aciertos += 1
            else:
                print(f"Error: Real='{clase_real}' vs Predicción='{prediccion}' (Datos: {elemento_a_probar})")

        # 6. Resultados Finales
        efectividad = (aciertos / total_muestras_prueba) * 100
        print("-" * 30)
        print(f"Muestras evaluadas (Test): {total_muestras_prueba}")
        print(f"Aciertos: {aciertos}")
        print(f"Errores: {total_muestras_prueba - aciertos}")
        print(f"EFECTIVIDAD REAL DEL ALGORITMO: {efectividad:.2f}%")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: No encuentro el archivo '{archivo}'.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")