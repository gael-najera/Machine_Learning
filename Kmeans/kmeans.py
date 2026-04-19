import numpy as np
import pandas as pd


class AgrupadorKMeans:
    def __init__(self, k=3, max_iteraciones=100):
        self.k = k
        self.max_iteraciones = max_iteraciones
        self.centroides = None

    def entrenar(self, X):
        X = np.array(X)

        # 1. SELECCIÓN ALEATORIA DE CENTROIDES
        np.random.seed(42)
        indices_aleatorios = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroides = np.copy(X[indices_aleatorios])

        for iteracion in range(self.max_iteraciones):
            # 2. ASIGNACIÓN
            etiquetas_grupos = self._asignar_grupos(X)
            centroides_anteriores = np.copy(self.centroides)

            # 3. ACTUALIZACIÓN (Cálculo del nuevo promedio)
            for i in range(self.k):
                puntos_del_grupo = X[etiquetas_grupos == i]
                if len(puntos_del_grupo) > 0:
                    self.centroides[i] = np.mean(puntos_del_grupo, axis=0)

            # 4. CONVERGENCIA
            if np.all(self.centroides == centroides_anteriores):
                print(f"El algoritmo convergió en la iteración {iteracion + 1}")
                break

    def _asignar_grupos(self, X):
        etiquetas = []
        for elemento in X:
            distancias = [np.linalg.norm(elemento - c) for c in self.centroides]
            etiquetas.append(np.argmin(distancias))
        return np.array(etiquetas)

    def predecir(self, X):
        return self._asignar_grupos(np.array(X))


if __name__ == "__main__":
    archivo = 'iris.csv'  # Se puede cambiar por cualquier otro archivo .csv

    try:
        df = pd.read_csv(archivo)
        print(f"Dataset cargado: {archivo} ({df.shape[0]} muestras)")

        # Separar X (datos) e Y (clases reales para validación)
        X = df.iloc[:, :-1].values
        y_real = df.iloc[:, -1].values

        # Configuración del modelo
        k_clusters = 3
        print(f"\nK-Means con K={k_clusters}")
        kmeans = AgrupadorKMeans(k=k_clusters)

        #Entrenar el modelo
        kmeans.entrenar(X)

        # Resultados finales
        y_predicciones = kmeans.predecir(X)

        print("\nMATRIZ DE CONFUSIÓN (Grupos vs Etiquetas Reales)")
        matriz = pd.crosstab(pd.Series(y_real, name='Clase Real'),
                             pd.Series(y_predicciones, name='Clúster Asignado'))
        print(matriz)

    except Exception as e:
        print(f"Error al procesar el dataset: {e}")