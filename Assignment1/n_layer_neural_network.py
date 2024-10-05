import numpy as np
import matplotlib.pyplot as plt

from Assignment1.three_layer_neural_network import plot_decision_boundary, generate_data


class Layer:
    def __init__(self, input_dim, output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        np.random.seed(seed)
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros((1, output_dim))
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

    def actFun(self, z, type):
        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif type == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Invalid activation function type: {type}")

    def diff_actFun(self, z, type):
        if type == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif type == 'sigmoid':
            return self.actFun(z, 'sigmoid') * (1 - self.actFun(z, 'sigmoid'))
        elif type == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError(f"Invalid activation function type: {type}")

    def feedforward(self, X):
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z, self.actFun_type)
        return self.a

    def backprop(self, da, prev_a):
        dz = da * self.diff_actFun(self.z, self.actFun_type)
        dW = np.dot(prev_a.T, dz) + self.reg_lambda * self.W
        db = np.sum(dz, axis=0, keepdims=True)
        da_prev = np.dot(dz, self.W.T)
        return da_prev, dW, db


class DeepNeuralNetwork:
    def __init__(self, nn_input_dim, nn_output_dim, hidden_layers, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.layers = []
        self.reg_lambda = reg_lambda
        self.actFun_type = actFun_type

        # Create input layer
        input_dim = nn_input_dim
        for hidden_dim in hidden_layers:
            self.layers.append(Layer(input_dim, hidden_dim, actFun_type, reg_lambda, seed))
            input_dim = hidden_dim

        # Create output layer
        self.layers.append(Layer(input_dim, nn_output_dim, actFun_type, reg_lambda, seed))

    def feedforward(self, X):
        a = X
        for layer in self.layers:
            a = layer.feedforward(a)
        exp_scores = np.exp(a)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)
        correct_logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)

        # L2 regularization
        for layer in self.layers:
            data_loss += self.reg_lambda / 2 * np.sum(np.square(layer.W))

        return (1. / num_examples) * data_loss

    def backprop(self, X, y):
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1

        # Backprop through each layer (from last to first)
        da = delta3
        for i in reversed(range(len(self.layers))):
            if i == 0:
                prev_a = X
            else:
                prev_a = self.layers[i - 1].a
            da, dW, db = self.layers[i].backprop(da, prev_a)

            # Gradient descent update
            self.layers[i].W += -0.01 * dW
            self.layers[i].b += -0.01 * db

    def fit_model(self, X, y, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(X, y)

            if print_loss and i % 1000 == 0:
                print(f"Loss after iteration {i}: {self.calculate_loss(X, y)}")

    def visualize_decision_boundary(self, X, y):
        plot_decision_boundary(lambda x: np.argmax(self.feedforward(x), axis=1), X, y)



def main():
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    # Define a deeper network with two hidden layers
    hidden_layers = [4, 5]
    model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2, hidden_layers=hidden_layers, actFun_type='tanh')
    model.fit_model(X, y)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()

