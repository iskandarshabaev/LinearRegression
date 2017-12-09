import numpy as np

# x = np.array([1, 2, 4, 3, 5])
# y = np.array([1, 3, 3, 2, 5])

x_train = np.arange(1, 5)
y_train = np.arange(1, 5)


class LM:
    b0 = 0.0
    b1 = 0.0

    X = []
    Y = []

    def fit(self, x, y):
        self.X = x
        self.Y = y

    def train(self, learning_rate=0.01, epochs=1000):
        for i in range(0, epochs):
            self.gradient_step(learning_rate, i)

    def predict(self, x):
        return self.b0 + self.b1 * x

    def error(self, x, y):
        return self.predict(x) - y

    def gradient_step(self, alpha, step):
        for i in range(0, len(self.X)):
            temp_b0 = self.b0 - alpha * self.error(self.X[i], self.Y[i])
            temp_b1 = self.b1 - alpha * self.error(self.X[i], self.Y[i]) * self.X[i]
            self.b0 = temp_b0
            self.b1 = temp_b1


lm = LM()
lm.fit(x_train, y_train)
lm.train(epochs=10000)
print(lm.b0, lm.b1)
print(lm.predict(x_train))
