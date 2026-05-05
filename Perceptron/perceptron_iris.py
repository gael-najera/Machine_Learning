import pandas as pd
import numpy as np


def entrenar_perceptron_general(ruta_archivo):
    # 1. Leer el archivo (Soporta Excel o CSV)
    try:
        if ruta_archivo.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(ruta_archivo)
        else:
            df = pd.read_csv(ruta_archivo)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return

    # Asegurarnos de que el Perceptrón solo trabaje con valores numéricos
    # Esto elimina columnas de texto (como 'class') que causan el error al multiplicar en np.dot
    df = df.select_dtypes(include=[np.number])

    # 2. Separar dinámicamente las entradas (X) del objetivo (y)
    # Asumimos que la ÚLTIMA columna es la variable a predecir (y_real)
    columna_objetivo = df.columns[-1]
    columnas_entrada = df.columns[:-1]

    # Extraer los datos en matrices de numpy
    X = df[columnas_entrada].values
    y = df[columna_objetivo].values

    total_registros, num_caracteristicas = X.shape

    # 3. Parámetros iniciales adaptables
    tasa_aprendizaje = 0.1
    # Inicializamos una lista de pesos (w) en ceros, del tamaño exacto de las entradas
    w = np.zeros(num_caracteristicas)
    b = 0.0

    epoca = 1
    max_epocas = 200
    errores_totales = -1

    print("\n" + "=" * 50)
    print("--- INICIANDO ENTRENAMIENTO GENERAL ---")
    print(f"Total de registros : {total_registros}")
    print(f"Columna objetivo   : '{columna_objetivo}'")
    print(f"Variables de input : {num_caracteristicas} {list(columnas_entrada)}")
    print("=" * 50 + "\n")

    # 4. Ciclo de entrenamiento
    while errores_totales != 0 and epoca <= max_epocas:
        errores_epoca = 0

        # Iterar sobre cada fila de datos
        for i in range(total_registros):
            x_actual = X[i]
            y_real = y[i]

            # Cálculo de la Suma (z). np.dot multiplica cada 'x' por su 'w' y los suma todos.
            z = np.dot(x_actual, w) + b
            y_hat = 1 if z >= 0 else 0

            error = y_real - y_hat

            # Actualizar todos los pesos dinámicamente
            if error != 0:
                # w se actualiza multiplicando toda la lista x_actual por el factor de aprendizaje
                w += tasa_aprendizaje * error * x_actual
                b += tasa_aprendizaje * error
                errores_epoca += 1

        errores_totales = errores_epoca
        efectividad = ((total_registros - errores_totales) / total_registros) * 100

        print(f"Vuelta {epoca} | Errores: {errores_totales} | Efectividad: {efectividad:.2f}%")

        if errores_totales == 0:
            print("\n" + "*" * 50)
            print("¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            print(f"-> Convergencia en la vuelta: {epoca}")
            print(f"-> Efectividad final: {efectividad:.2f}%")
            # Mostrar los pesos finales redondeados convirtiendo el tipo de numpy a float nativo de Python
            pesos_formateados = [round(float(peso), 4) for peso in w]
            print(f"-> Pesos finales (w): {pesos_formateados}")
            print(f"-> Bias final (b): {b:.4f}")
            print("*" * 50 + "\n")
            break

        epoca += 1

    if epoca > max_epocas:
        efectividad = ((total_registros - errores_totales) / total_registros) * 100
        print("\n" + "!" * 50)
        print("¡LÍMITE DE VUELTAS ALCANZADO!")
        print(f"-> El modelo se detuvo sin converger a cero errores.")
        print(f"-> Efectividad final estancada en: {efectividad:.2f}%")
        print("!" * 50 + "\n")


# Solo le pasas el archivo, ¡y el código se encarga de entenderlo!
entrenar_perceptron_general('Iris2.xlsx')