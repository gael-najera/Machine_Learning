import numpy as np
import pandas as pd


class ClasificadorNaiveBayes:
    def __init__(self):
        self.clases = None
        self.medias = {}
        self.varianzas = {}
        self.priors = {}

    def entrenar(self, X, y):
        self.clases = np.unique(y)
        n_muestras_totales = len(y)

        for clase in self.clases:
            X_clase = X[y == clase]
            self.medias[clase] = np.mean(X_clase, axis=0)
            self.varianzas[clase] = np.var(X_clase, axis=0)
            self.priors[clase] = len(X_clase) / n_muestras_totales

    def _probabilidad_gaussiana(self, x, media, varianza):
        eps = 1e-6
        exponente = np.exp(-((x - media) ** 2) / (2 * (varianza + eps)))
        return (1 / np.sqrt(2 * np.pi * (varianza + eps))) * exponente

    def clasificar(self, elemento):
        elemento = np.array(elemento)
        probabilidades_finales = []

        for clase in self.clases:
            prior = np.log(self.priors[clase])
            probabilidad_condicional = np.sum(
                np.log(self._probabilidad_gaussiana(elemento, self.medias[clase], self.varianzas[clase])))
            posterior = prior + probabilidad_condicional
            probabilidades_finales.append((posterior, clase))

        probabilidades_finales.sort(key=lambda x: x[0], reverse=True)
        return probabilidades_finales[0][1]


if __name__ == "__main__":
    archivo = 'iris.csv'

    try:
        df = pd.read_csv(archivo)
        print(f"Archivo '{archivo}' cargado. Dimensiones: {df.shape}")

        columna_clase = df.columns[-1]

        muestras_train = df.groupby(columna_clase, group_keys=False).apply(
            lambda x: x.sample(frac=0.8, random_state=42))
        indices_train = muestras_train.index

        df_train = df.loc[indices_train]
        df_test = df.drop(indices_train)

        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values
        X_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values
        indices_test = df_test.index.tolist()  # ← CAMBIO 1: guardar índices originales del CSV

        clf = ClasificadorNaiveBayes()
        clf.entrenar(X_train, y_train)
        print("\nModelo Naive Bayes entrenado")

        print("\nIniciando clasificación")
        aciertos = 0
        total_muestras_prueba = len(X_test)

        y_reales = []
        y_predicciones = []

        for i in range(total_muestras_prueba):
            elemento_a_probar = X_test[i]
            clase_real = y_test[i]

            prediccion = clf.clasificar(elemento_a_probar)

            y_reales.append(clase_real)
            y_predicciones.append(prediccion)

            if prediccion == clase_real:
                aciertos += 1
            else:
                # ← CAMBIO 2: mostrar el índice original del CSV
                print(f"Error [índice CSV {indices_test[i]}]: Real='{clase_real}' vs Pred='{prediccion}' (Datos: {elemento_a_probar})")

        efectividad = (aciertos / total_muestras_prueba) * 100

        print("\n" + "=" * 35)
        print("      RESULTADOS FINALES")
        print("=" * 35)
        print(f"Muestras evaluadas: {total_muestras_prueba}")
        print(f"Aciertos: {aciertos}")
        print(f"Errores: {total_muestras_prueba - aciertos}")
        print(f"EFECTIVIDAD: {efectividad:.2f}%")

        print("\n--- Matriz de Confusión ---")
        matriz = pd.crosstab(pd.Series(y_reales, name='Real'), pd.Series(y_predicciones, name='Predicción'))
        print(matriz)
        print("=" * 35)

    except FileNotFoundError:
        print(f"Error: No encuentro el archivo '{archivo}'.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")