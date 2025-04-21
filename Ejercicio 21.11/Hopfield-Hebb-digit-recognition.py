from matplotlib import pyplot as plt
import numpy as np

from digit_patterns import digits
from digit_patterns_corrupted import digits_corrupted

# Ejercicio 21.11 i del capitulo 21 de Neural Networks design

def calcular_matriz_pesos(patrones, n):
    # Inicializar la matriz de pesos
    W = np.zeros((n, n))

    # Calcular la matriz de pesos
    for p in patrones:
        W += np.outer(p, p)
    # Poner los elementos diagonales ya que las neuronas no se conectan entre sí
    np.fill_diagonal(W, 0)

    # normalizar la matriz de pesos
    W /= len(patrones)
    return W

def calcular_energia(patron, W):
    # Calcular la energía del patrón dado la matriz de pesos
    return -0.5 * np.dot(patron, np.dot(W, patron))

def correr_algoritmo_hopfield_asincronico(patron_inicial, W, num_iteraciones):
    # Inicializar el patrón actual
    patron_actual = np.copy(patron_inicial)
    energia = calcular_energia(patron_actual, W)

    # Iterar para actualizar el patrón
    for _ in range(num_iteraciones):
        # recorrer todas las neuronas en orden aleatorio
            # crear lista de indices aleatorios
        indices = np.random.permutation(len(patron_actual))
        for i in indices:
            # Calcular la nueva activación de la neurona
            patron_actual[i] = np.sign(W[i].T @ patron_actual)

        # calcular la energía del patrón actual
        energia_nueva = calcular_energia(patron_actual, W)
        # Si la energía no cambio, se puede salir del bucle (convergió el patrón)
        if energia == energia_nueva:
            print("Converge en iteracion: ", _)
            return patron_actual
        else:
            energia = energia_nueva
    

    return patron_actual

def main():
    
    # cant patrones
    k = 10
    # tamaño de la red
    n = len(digits[0].flatten())

    pesos = calcular_matriz_pesos(digits.values(), n)
    predicciones = [correr_algoritmo_hopfield_asincronico(patron.flatten(), pesos, num_iteraciones=100000).reshape(6, 5) for patron in digits.values()]

    # muestro los patrones aleatorios y sus predicciones
    fig, axs = plt.subplots(k, 2, figsize=(10, 20))
    for i in range(k):
        axs[i, 0].imshow(digits[i])
        axs[i, 1].imshow(predicciones[i])
        # sacar el eje x e y
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    
    axs[0, 0].set_title('Patrón Aleatorio')
    axs[0, 1].set_title('Predicción')

    plt.show()

if __name__ == "__main__":
    main()