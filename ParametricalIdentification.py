import csv
import numpy as np

# Create a CSV file with a header to store optimization results
def create_csv(filename):
    headers = ['iteration', 'a', 'b', 'c', 'quality']  # CSV column names
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write header to CSV file

# Append data (a list of rows) to the existing CSV file
def append_to_csv(filename, data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)  # Write rows of data to file

# Define a linear model function: y = ax + b
def linear_func(x, a, b):
    return a * x + b

# Define an exponential model function: y = a * exp(bx) + c
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Define a power-based polynomial function: y = a * x^b + c
def polynomial_func(x, a, b, c):
    return a * x**b + c

# Compute the analytical gradients of the cost function (MSE) w.r.t. parameters
def compute_analytical_gradients(x, y, y_pred, params, function_type):
    n = len(y)  # Number of data points
    
    # Linear model gradient: y = ax + b
    if function_type == 'linear':
        a, b = params
        a_grad = -2 * np.sum(x * (y - y_pred)) / n
        b_grad = -2 * np.sum(y - y_pred) / n
        return np.array([a_grad, b_grad])
    
    # Exponential model gradient: y = a * exp(bx) + c
    elif function_type == 'exponent':
        a, b, c = params
        a_grad = -2 * np.sum(np.exp(b * x) * (y - y_pred)) / n
        b_grad = -2 * np.sum(a * x * np.exp(b * x) * (y - y_pred)) / n
        c_grad = -2 * np.sum(y - y_pred) / n
        return np.array([a_grad, b_grad, c_grad])
    
    # Polynomial model gradient: y = a * x^b + c
    elif function_type == 'polynomial':
        # Avoid division by zero and log(0) by replacing zero x values with a small number
        x = np.where(x == 0, 1e-6, x)
        a, b, c = params
        a_grad = -2 * np.sum(x**b * (y - y_pred)) / n
        b_grad = -2 * np.sum(a * x**b * np.log(x) * (y - y_pred)) / n
        c_grad = -2 * np.sum(y - y_pred) / n
        return np.array([a_grad, b_grad, c_grad])

# Generalized function to perform least squares fitting with Adam optimizer
def general_least_squares_fit(x, y, func, params, function_type,
                              epsilon=1e-5, filename="result.csv", max_iter=30000, is_print=False):
    """
    Fits a specified function to data using least squares optimization with the Adam optimizer.

    Parameters:
    -----------
    x : array-like
        The input (independent variable) data points.

    y : array-like
        The observed (dependent variable) data points corresponding to `x`.

    func : callable
        The model function to fit, e.g., linear_func, exponential_func, etc.
        It should accept x and the model parameters as arguments.

    params : np.ndarray
        Initial guess for the parameters to be optimized (e.g., [a, b, c]).

    function_type : str
        A string identifier for the function type: "linear", "exponent", or "polynomial".
        Determines which analytical gradients to compute.

    epsilon : float, optional (default=1e-5)
        Convergence threshold. Optimization stops when the change in error between
        iterations is smaller than this value.

    filename : str, optional (default="result.csv")
        Path to the CSV file used to log the optimization progress at each iteration.

    max_iter : int, optional (default=30000)
        The maximum number of iterations to perform before stopping, regardless of convergence.

    is_print : bool, optional (default=False)
        If True, prints the current iteration, parameters, and error (fit quality) to stdout.

    Returns:
    --------
    params : np.ndarray
        The optimized parameters for the model function.
    """

    create_csv(filename)  # Initialize result file

    n = len(y)  # Number of data points
    iter_num = 0  # Iteration counter
    quality = float('inf')  # Previous error value (initialized high)
    learning_rate = 0.01  # Step size for gradient descent
    new_quality = 0  # Current error value
    
    # Adam optimizer hyperparameters
    beta1 = 0.9      # Exponential decay rate for the first moment (mean)
    beta2 = 0.999    # Exponential decay rate for the second moment (variance)
    epsilon_adam = 1e-8  # Small constant to avoid division by zero
    m = np.zeros_like(params)  # First moment vector
    v = np.zeros_like(params)  # Second moment vector
    
    # Main optimization loop
    while abs(quality - new_quality) > epsilon:
        iter_num += 1  # Increment iteration counter

        # Calculate current model predictions
        y_pred = func(x, *params)

        # Compute residuals (errors)
        residuals = y - y_pred
        quality = new_quality  # Update previous quality

        # Compute current fit quality (sum of squared residuals)
        new_quality = np.sum(residuals ** 2)

        # Compute parameter gradients analytically for the current model
        grads = compute_analytical_gradients(x, y, y_pred, params, function_type)

        # Adam optimizer update rule
        m = beta1 * m + (1 - beta1) * grads  # Update biased first moment estimate
        v = beta2 * v + (1 - beta2) * (grads ** 2)  # Update biased second moment estimate
        m_hat = m / (1 - beta1 ** iter_num)  # Bias-corrected first moment estimate
        v_hat = v / (1 - beta2 ** iter_num)  # Bias-corrected second moment estimate

        # Update parameters using Adam step
        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon_adam)

        # Optionally print current state
        if is_print:
            print(f"iter: {iter_num}, params: {params}, quality: {quality}")

        # Log the iteration results to CSV
        append_to_csv(filename, [[iter_num] + list(params) + [quality]])

        # Stop if maximum number of iterations is exceeded
        if iter_num > max_iter:
            return params

    # Return the optimized parameters once convergence is achieved
    return params
