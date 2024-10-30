import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLP_PSO:
    def __init__(self, input_size, hidden_layers, output_size, particle_size, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.particle_size = particle_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

    def relu(self, x):
        return np.maximum(0, x)

    def initialize_particles(self):
        weights_shape = []
        input_dim = self.input_size
        for hidden in self.hidden_layers:
            weights_shape.append((input_dim, hidden))
            input_dim = hidden
        weights_shape.append((input_dim, self.output_size))

        particles = [self.random_weights(weights_shape) for _ in range(self.particle_size)]
        velocities = [self.random_weights(weights_shape) for _ in range(self.particle_size)]
        return particles, velocities, weights_shape

    def random_weights(self, shapes):
        return [np.random.uniform(-1, 1, size=shape) for shape in shapes]

    def forward_pass(self, x, weights):
        for w in weights[:-1]:
            x = self.relu(np.dot(x, w))
        return np.dot(x, weights[-1])

    def predict(self, x, weights):
        return self.forward_pass(x, weights)

    def evaluate(self, x, y, weights):
        predictions = self.predict(x, weights)
        mae = np.mean(np.abs(predictions - y))
        return mae, predictions

    def train(self, x_train, y_train):
        particles, velocities, shapes = self.initialize_particles()
        pbest = particles.copy()
        gbest = min(particles, key=lambda p: self.evaluate(x_train, y_train, p)[0])
        gbest_mae = self.evaluate(x_train, y_train, gbest)[0]

        for iter in range(self.max_iter):
            for i in range(self.particle_size):
                velocities[i] = [self.w * velocities[i][j] +
                                 self.c1 * np.random.rand() * (pbest[i][j] - particles[i][j]) +
                                 self.c2 * np.random.rand() * (gbest[j] - particles[i][j])
                                 for j in range(len(shapes))]

                particles[i] = [particles[i][j] + velocities[i][j] for j in range(len(shapes))]

                if self.evaluate(x_train, y_train, particles[i])[0] < self.evaluate(x_train, y_train, pbest[i])[0]:
                    pbest[i] = particles[i]

            current_gbest = min(particles, key=lambda p: self.evaluate(x_train, y_train, p)[0])
            current_gbest_mae = self.evaluate(x_train, y_train, current_gbest)[0]

            if current_gbest_mae < gbest_mae:
                gbest = current_gbest
                gbest_mae = current_gbest_mae

            print(f"Iteration {iter + 1}/{self.max_iter}, Best MAE: {gbest_mae}")
        return gbest


class Dataset:
    def __init__(self, file_path, predict_days=5):
        self.file_path = file_path
        self.predict_days = predict_days

    def prepare_data(self):
        df = pd.read_excel(self.file_path)
        x = df.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values
        y = df.iloc[:, 5].shift(-self.predict_days).dropna().values
        x = x[:-self.predict_days]
        return x, y

class CrossValidator:
    def __init__(self, model, k_folds=10):
        self.model = model
        self.k_folds = k_folds

    def cross_validation(self, x, y):
        fold_size = len(x) // self.k_folds
        mae_scores = []
        all_predictions = []

        for i in range(self.k_folds):
            x_train = np.concatenate((x[:i * fold_size], x[(i + 1) * fold_size:]), axis=0)
            y_train = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]), axis=0)
            x_test = x[i * fold_size:(i + 1) * fold_size]
            y_test = y[i * fold_size:(i + 1) * fold_size]

            best_weights = self.model.train(x_train, y_train)
            mae, predictions = self.model.evaluate(x_test, y_test, best_weights)
            mae_scores.append(mae)
            all_predictions.append((y_test, predictions))
            print(f"Fold {i + 1}, MAE: {mae}")

        avg_mae = np.mean(mae_scores)
        print(f"Average MAE over {self.k_folds} folds: {avg_mae}")

        self.plot_mae(mae_scores)
        self.plot_true_vs_pred(all_predictions[0][0], all_predictions[0][1])

    def plot_mae(self, mae_scores):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.k_folds + 1), mae_scores, marker='o')
        plt.title("Mean Absolute Error for Each Fold")
        plt.xlabel("Fold")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.show()

    def plot_true_vs_pred(self, y_true, y_pred):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='True Values', color='b')
        plt.plot(y_pred, label='Predicted Values', color='r', linestyle='--')
        plt.title("True vs Predicted Values")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    dataset = Dataset('C:/Users/Admin/MLP swarm/AirQualityUCI.xlsx', predict_days=5)
    x, y = dataset.prepare_data()

    input_size = x.shape[1]
    output_size = 1
    
    particle_size = 3
    max_iter = 30
    hidden_layers = [30, 10]
    
    mlp_pso = MLP_PSO(input_size, hidden_layers, output_size, particle_size, max_iter)

    cross_validator = CrossValidator(mlp_pso, k_folds=10)
    cross_validator.cross_validation(x, y)
