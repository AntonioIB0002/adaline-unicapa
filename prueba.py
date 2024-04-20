import matplotlib.pyplot as plt
import numpy as np

def plano_cartesiano(coordenadas):
    
    x_coords, y_coords = zip(*coordenadas)
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')

    ax.scatter(x_coords, y_coords, color='r')
    plt.grid(True)
    # Guardar la imagen en un archivo
    fig.savefig('plano_actualizado.png')
    filename = 'plano_actualizado.png'
    return filename

def predict(X, W, b):
    y = np.dot(X, W) + b
    return activation_function(y)
def activation_function(z):
    return 1 / (1 + np.exp(-z)) 
# Función para graficar contornos
def plot_contour(X, W,b,y):
    plt.figure(figsize = (8,8))

    # Crear una cuadrícula de puntos en el rango de tus datos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predecir las clases para cada punto en la cuadrícula
    Z = np.argmax(predict(np.c_[xx.ravel(), yy.ravel()], W, b), axis=1)
    Z = Z.reshape(xx.shape)
    
    # Graficar los contornos
    plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y,cmap=plt.cm.coolwarm, marker='o', edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Contorno de las clases')
    # Mostrar gráfico
    
    filename = 'plano_actualizado.png'
    plt.savefig(filename)
    return filename

# np.random.seed(0)
# num_samples = 4
# X = np.random.randn(4, 2)  # Dos dimensiones (x, y)
# y = [8,4,2,0]
# print(y) # Clases: 0, 1, 2, 3 (círculos, triángulos, estrellas, cuadrados)
# # Supongamos que tienes tus pesos (W) y el sesgo (b) de tu red neuronal entrenada
# # Aquí utilizaremos valores aleatorios para propósitos de demostración
# W = np.random.randn(2, 4)  # Pesos
# b = np.random.randn(4)
# plot_contour(X,W,b,y)