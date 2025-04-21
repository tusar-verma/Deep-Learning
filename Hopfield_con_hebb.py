import numpy as np

# Ejercicio 21.1 i del capitulo 21 de Neural Networks design
# consiste en implementar una red de Hopfield con memoria de 2 patrones usando la regla de Hebb

def calcular_matriz_pesos(patrones, n, k):
    # Inicializar la matriz de pesos
    W = np.zeros((n, n))

    # Calcular la matriz de pesos
    for p in patrones:
        W += np.outer(p, p)

    # Poner los elementos diagonales ya que las neuronas no se conectan entre sí
    np.fill_diagonal(W, 0)

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
            return patron_actual
        else:
            energia = energia_nueva
    

    return patron_actual

def main():
    
    # cant patrones
    k = 2
    # tamaño de la red
    n = 4
    # patrones de entrada
    patrones = [np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1])]

    pesos = calcular_matriz_pesos(patrones, n, k)

    print(correr_algoritmo_hopfield_asincronico(np.array([-1, 1, -1, 1]), pesos, num_iteraciones=10))

if __name__ == "__main__":
    main()