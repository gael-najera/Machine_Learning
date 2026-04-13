import numpy as np
import pandas as pd


class ClasificadorMinimaDistancia:
    def __init__(self):
        self.representantes = {}

    def entrenar(self, X, y):
        # Aseguramos formato numpy
        X = np.array(X)
        y = np.array(y)
        self.clases_unicas = np.unique(y)

        for clase in self.clases_unicas:
            instancias = X[y == clase]
            # Calculamos el centroide (media) de esta clase
            self.representantes[clase] = np.mean(instancias, axis=0)

    def clasificar(self, elemento):
        elemento = np.array(elemento)
        min_dist = float('inf')
        clase_predicha = None

        for clase, centroide in self.representantes.items():
            dist = np.linalg.norm(elemento - centroide)
            if dist < min_dist:
                min_dist = dist
                clase_predicha = clase
        return clase_predicha


# CÁLCULO DE EFECTIVIDAD
if __name__ == "__main__":
    archivo = 'iris.csv'  # Nombre del archivo

    try:
        # 1. Cargar datos
        df = pd.read_csv(archivo)
        print(f"Archivo '{archivo}' cargado. Dimensiones: {df.shape}")

        # 2. Separar universalmente (clase en la última columna)
        X = df.iloc[:, :-1].values  # Características (Números)
        y = df.iloc[:, -1].values  # Etiquetas

        # 3. Entrenar el modelo
        clf = ClasificadorMinimaDistancia()
        clf.entrenar(X, y)
        print("\n--- Centroides calculados ---")
        for k, v in clf.representantes.items():
            print(f"Clase '{k}': {v}")

        # 4. PRUEBA DE EFECTIVIDAD (Elemento por elemento)
        print("\n--- Iniciando prueba de clasificación ---")
        aciertos = 0
        total_muestras = len(X)

        for i in range(total_muestras):
            # Tomamos las características del elemento 'i'
            elemento_a_probar = X[i]
            # Tomamos la etiqueta real del elemento 'i'
            clase_real = y[i]

            # El algoritmo predice
            prediccion = clf.clasificar(elemento_a_probar)

            # Comparamos
            if prediccion == clase_real:
                aciertos += 1
            else:
                # Imprimir errores para analizar
                print(f"Error en índice {i}: Real={clase_real} vs Predicción={prediccion}")

        # 5. Resultados Finales
        efectividad = (aciertos / total_muestras) * 100
        print("-" * 30)
        print(f"Total de muestras: {total_muestras}")
        print(f"Aciertos: {aciertos}")
        print(f"Errores: {total_muestras - aciertos}")
        print(f"EFECTIVIDAD DEL ALGORITMO: {efectividad:.2f}%")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: No encuentro el archivo '{archivo}'.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")