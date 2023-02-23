# Import JAX's version of NumPy
import numpy as np 

from njax.layers import Dense
from njax.activations import Tanh
from njax.losses.losses import mse, mse_prime
from njax.network import train, predict

# XOR dataset
X = np.reshape([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], (4, 2, 1))
Y = np.reshape([[0.0], [1.0], [1.0], [0.0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# train
train(network, mse, mse_prime, X, Y, epochs=100, learning_rate=0.1)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)
print(f"The outputs are: {points}")

# if you want to plot the decision boundary in 3D
# when you run this script, uncomment the following lines
# and install matplotlib
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
#     plt.show()