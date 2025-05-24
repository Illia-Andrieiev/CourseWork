import numpy as np
import csv

def create_csv(filename):
    headers = ['iteration', 'a', 'b', 'c', 'quality']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def append_to_csv(filename, data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def linear_func(x, a, b):
    return a * x + b

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def polynomial_func(x, a, b, c):
    return a * x**b + c

def compute_analytical_gradients(x, y, y_pred, params, function_type):
    n = len(y)
    
    if function_type == 'linear':
        a, b = params
        a_grad = -2 * np.sum(x * (y - y_pred)) / n
        b_grad = -2 * np.sum(y - y_pred) / n
        return np.array([a_grad, b_grad])
    
    elif function_type == 'exponent':
        a, b, c = params
        a_grad = -2 * np.sum(np.exp(b * x) * (y - y_pred)) / n
        b_grad = -2 * np.sum(a * x * np.exp(b * x) * (y - y_pred)) / n
        c_grad = -2 * np.sum(y - y_pred) / n
        return np.array([a_grad, b_grad, c_grad])
    
    elif function_type == 'polynomial':
        x = np.where(x == 0, 1e-6, x)  
        a, b, c = params
        a_grad = -2 * np.sum(x**b * (y - y_pred)) / n
        b_grad = -2 * np.sum(a * x**b * np.log(x) * (y - y_pred)) / n
        c_grad = -2 * np.sum(y - y_pred) / n
        return np.array([a_grad, b_grad, c_grad])

def general_least_squares_fit(x, y, func, params, function_type, epsilon=1e-5, filename="result.csv", max_iter = 30_000, is_print=False):
    create_csv(filename)
    
    n = len(y)
    iter_num = 0
    quality = float('inf')
    learning_rate = 0.01
    new_quality = 0
    
    # Параметры Адам
    beta1 = 0.9
    beta2 = 0.999
    epsilon_adam = 1e-8
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    
    while abs(quality - new_quality) > epsilon:
        iter_num += 1
        
        # Предсказания
        y_pred = func(x, *params)
        
        # Ошибки
        residuals = y - y_pred
        quality = new_quality
        
        # Качество соответствия (сумма квадратов ошибок)
        new_quality = np.sum(residuals ** 2)
        
        # Вычисление градиентов аналитически
        grads = compute_analytical_gradients(x, y, y_pred, params, function_type)
        
        # Адам обновление параметров
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** iter_num)
        v_hat = v / (1 - beta2 ** iter_num)
        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)
        
        if is_print:
            print(f"iter: {iter_num}, params: {params}, quality: {quality}")
        
        # Сохранение результатов на каждой итерации
        append_to_csv(filename, [[iter_num] + list(params) + [quality]])
        if iter_num > max_iter:
            return params
    return params
