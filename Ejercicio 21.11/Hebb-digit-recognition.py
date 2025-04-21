import numpy as np
import matplotlib.pyplot as plt

from digit_patterns import digits
from digit_patterns_corrupted import digits_corrupted

def calcular_matriz_pesos_hasta_patron_k(patrones, n, k):
    # Inicializar la matriz de pesos
    W = np.zeros((n, n))

    # Calcular la matriz de pesos
    for i in range(k):
        p = patrones[i].flatten()
        W += np.outer(p, p)

    return W

def predecir_hebb(patron, W):
    # Calcular la activación de la neurona
    return np.sign(W @ patron)



def main():
    # cantidad de neuronas
    n = len(digits[0].flatten())
    # limite de patrones
    k =10
    # Matriz de pesos con patrones 0 y 1
    W = calcular_matriz_pesos_hasta_patron_k(digits, n, k)


    # obtengo las predicciones de los patrones aleatorios
    predicciones = [predecir_hebb(patron.flatten(), W).reshape(6, 5) for patron in digits.values()]
    
    # muestro los patrones aleatorios y sus predicciones
    fig, axs = plt.subplots(k, 2, figsize=(10, 20))
    for i in range(k):
        axs[i, 0].imshow(digits[i])
        axs[i, 1].imshow(predicciones[i])
        # sacar el eje x e y
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

    
    axs[0, 0].set_title('Patrón Aleatorio')
    axs[0, 1].set_title('Predicción Hebb')

    plt.show()
    


if __name__ == "__main__":
    main()