def entrenar_compuerta(nombre, X, Y, tasa_aprendizaje=0.1, max_epocas=20):
    # Valores iniciales (idénticos a tu Excel)
    w1 = 0.5
    w2 = -0.5
    b = 0.1

    print(f"\n--- Entrenando Compuerta {nombre} ---")

    for epoca in range(1, max_epocas + 1):
        errores_epoca = 0

        for i in range(len(X)):
            x1, x2 = X[i]
            y_real = Y[i]

            # 1. Suma ponderada (z)
            z = (x1 * w1) + (x2 * w2) + b

            # 2. Predicción (y_hat) usando función escalón
            y_hat = 1 if z >= 0 else 0

            # 3. Cálculo del Error
            error = y_real - y_hat

            # 4. Actualización de pesos si hay error
            if error != 0:
                w1 = w1 + (tasa_aprendizaje * error * x1)
                w2 = w2 + (tasa_aprendizaje * error * x2)
                b = b + (tasa_aprendizaje * error)
                errores_epoca += 1

        # Verificar convergencia al final de cada época
        if errores_epoca == 0:
            print(f"¡Convergencia alcanzada en la época {epoca}!")
            print(f"Pesos finales: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")
            
            # Imprimir la tabla de verdad comprobando con los pesos finales
            print("\nTabla de Verdad Final:")
            print(" x1 | x2 | Esperado | Obtenido ")
            print("-" * 32)
            for i in range(len(X)):
                x1, x2 = X[i]
                y_real = Y[i]
                z_final = (x1 * w1) + (x2 * w2) + b
                y_hat_final = 1 if z_final >= 0 else 0
                print(f"  {x1} |  {x2} |    {y_real}     |    {y_hat_final}")
            print("-" * 32)
            break


# Datos de entrada (x1, x2)
X_datos = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Resultados esperados (y_real)
Y_and = [0, 0, 0, 1]
Y_or = [0, 1, 1, 1]

# Ejecutar entrenamiento
entrenar_compuerta("AND", X_datos, Y_and)
entrenar_compuerta("OR", X_datos, Y_or)