import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution

X = np.c_[np.ones((len(x_train), 1)), x_train]
# np.ones tworzy kolumne jedynek - reprezentuje wyraz wolny w modelu
# np.c - laczy kolumne jedynek z wektorem x_train, dostajemy macierz X(n,2), gdzie n to liczba probek

theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)
# implementacja wzoru theta = (X^T X)^-1 X^T y
# np.linalg.inv - obliczenie inwersji macierzy
# .dot - mnozenie macierzy
# .T - transponowanie macierzy

print("Theta: ", theta_best)

# TODO: calculate error
def mse(_theta, _x, _y):
    m = len(_x)
    # y_pred = iloczyn wektorow x i theta
    y_pred = _x.dot(_theta)
    # wzor na mse (1.3 instrukcja)
    return np.sum((y_pred - _y) ** 2) / m

X_test = np.c_[np.ones((len(x_test), 1)), x_test]
X_train = np.c_[np.ones((len(x_train), 1)), x_train]

print("MSE for train set: ", mse(theta_best, X_train, y_train))
print("MSE for test set: ", mse(theta_best, X_test, y_test))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# ze wzoru 1.15 instrukcja
# standaryzacja Z

#obliczenie odchylenia standardowego dla x i y (train)
x_standard_deviation = np.std(x_train)
y_standard_deviation = np.std(y_train)

# obliczenie sredniej dla x i y
x_average = np.average(x_train)
y_average = np.average(y_train)

# standaryzacja wedlug wzoru - dla probki odjac srednia i podzielic przez odchylenie
x_train_standardized = (x_train - x_average) / x_standard_deviation
y_train_standardized = (y_train - y_average) / y_standard_deviation
# do standaryzacji danych testowych uzyc tych samych statystyk (srednia i odchylenie)
# obliczonych na zbiorze treningowym - modele testowe sa przeskalowane w ten sam sposob
# co treningowe, co zapewnia uczciwa ocene modelu
x_test_standardized = (x_test - x_average) / x_standard_deviation
y_test_standardized = (y_test - y_average) / y_standard_deviation

# przygotowanie macierzy X i Y
# macierz X w pierwszej kolumnie ma jedynki, a w drugiej standaryzowane wartosci cechy
X_train_standardized = np.c_[np.ones((len(x_train_standardized), 1)), x_train_standardized]
X_test_standardized = np.c_[np.ones((len(x_test_standardized), 1)), x_test_standardized]
# reshape przeksztalca wektor Y w macierz kolumnowa
Y_train_standardized = y_train_standardized.reshape(-1, 1)
Y_test_standardized = y_test_standardized.reshape(-1, 1)

# TODO: calculate theta using Batch Gradient Descent

# theta to macierz 2x1, do ktorej wpisujemy losowe wartosci
theta = np.random.randn(2, 1)
# wspolczynnik uczenia
eta = 0.0001
max_epoch = 100000

# obliczenie gradientu bledu
def mse_gradient(_theta, _x, _y):
    m = len(_x)
    # wzor 1.7 instrukcja
    return 2/m * (_x.T.dot(_x.dot(_theta) - _y))


def calculate_theta_using_batch_gradient_descent(_theta, _x, _y, _eta):
    for i in range(max_epoch):
        # obliczenie gradientu bledu
        gradients = mse_gradient(_theta, _x, _y)
        # obliczenie thety (wzor 1.14)
        _theta = _theta - _eta * gradients
    return _theta

theta = calculate_theta_using_batch_gradient_descent(theta, X_train_standardized, Y_train_standardized, eta)
theta = theta.reshape(-1)

print('Theta:', theta)


# TODO: calculate error

# standaryzacja
x_standarized = (x_test - x_average) / x_standard_deviation
y_results = theta[1] * x_standarized + theta[0]
y_pred = (y_results * y_standard_deviation) + y_average

mse_bgd = sum((theta[0] + theta[1] * x_train_standardized - y_train_standardized)**2)/x_train_standardized.size
print(f"MSE for Batch Gradient Descent: {mse_bgd}")

# plot the regression line
x = np.linspace(min(x_test_standardized), max(x_test_standardized), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test_standardized, y_test_standardized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()