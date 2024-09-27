import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# PART 1: DISCRETE FOURIER TRANSFORM

file_path = 'f7.txt'

# Завантаження спостережень
observations = np.loadtxt(file_path)
y_obs = np.loadtxt('f7.txt', delimiter=' ')

T = 5
delta_t = 0.01
delta_f = 1 / T
N = len(observations)
n = np.arange(N)
k = n.reshape((N, 1))
e = np.exp(-2j * np.pi * k * n / N)

# Розрахунок частот
frequencies = [k * delta_t for k in range(1, len(observations))]
t = np.arange(0, T + delta_t, delta_t)

# Дискретне перетворення Фур'є (DFT)
DFT = np.dot(e, observations)
DFT_modules = np.abs(DFT)

# Функція для пошуку локальних екстремумів модуля DFT
def LocalExtremaOfDFTs(DFTM: np.ndarray):
    N = (len(DFTM) // 2)  # Працюємо лише з половиною спектру
    local_extrema_indices = []

    for i in range(1, N - 1):
        if DFTM[i] > DFTM[i - 1] and DFTM[i] > DFTM[i + 1]:
            local_extrema_indices.append(i)

    return local_extrema_indices

# Пошук локальних максимумів
local_extrema_indices = LocalExtremaOfDFTs(DFT_modules)

# Обчислення частот f_i на основі локальних максимумів
frequencies_of_extrema = [k * delta_f for k in local_extrema_indices]

# Виведення знайдених частот
for k in local_extrema_indices:
    freq = k * delta_f
    print(f"Local maximum at frequency {freq:.2f} Hz")

# Побудова графіку DFT
plt.plot(frequencies, DFT_modules[1:len(observations)])
plt.xlabel('Частота (Hz)')
plt.ylabel('Амплітуда')
plt.title('Discrete Fourier Transform (DFT)')
plt.grid()
plt.show()

############################################################################################
############################################################################################

# PART 2: LEAST SQUARES METHOD

# Відомі частоти f_i з дискретного перетворення Фур'є
f = frequencies_of_extrema

# Модель
def model(t, a, f):
    k = (len(f) + 3) # Кількість параметрів a_j
    model_value = a[0] * t**3 + a[1] * t**2 + a[2] * t
    for i in range(3, k):
        model_value += a[i] * np.sin(2 * np.pi * f[i-3] * t)
    model_value += a[k]  # Додатковий параметр a_k+1
    return model_value

# Функція похибки для методу найменших квадратів
def error_function(a, t, y_obs, f):
    y_pred = model(t, a, f)
    return np.sum((y_pred - y_obs)**2)  # Функціонал похибки

# Початкові наближення для параметрів a_j
initial_guess = np.ones(len(f) + 4)

# Мінімізація функціоналу похибки
result = minimize(error_function, initial_guess, args=(t, observations, f))

# Знайдені параметри a_j
a_opt = result.x

# Поміняти місцями останнє та передостанне значення
a_opt[-1], a_opt[-2] = a_opt[-2], a_opt[-1]

print(f"Знайдені параметри a_j (з поміненими місцями): {a_opt}")

# Побудова графіків для порівняння моделі та спостережень
y_pred = model(t, a_opt, f)

plt.figure(figsize=(10, 6))
plt.plot(t, observations, label="Спостереження", color='blue')
plt.plot(t, y_pred, label="Модель з отриманими параметрами", linestyle='--', color='red')
plt.xlabel('Час (t)')
plt.ylabel('y(t)')
plt.legend()
plt.title('Порівняння спостережень та моделі')
plt.grid()
plt.show()
