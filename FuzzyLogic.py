import numpy as np 
import matplotlib.pyplot as plt
import math

class FuzzyTerm:
    """
    Represents a fuzzy logic term with a name and a membership function.

    Attributes:
        _name (str): The name of the fuzzy term.
        _membership_function (function): A function that determines the degree of membership for a given input.
    """
    def __init__(self, name: str, membership_function):
        self._name = name
        self._membership_function = membership_function  # Membership function

    def get_name(self):
        """Returns the name of the fuzzy term."""
        return self._name

    def get_membership_value(self, x):
        """
        Evaluates the membership function for the given input.

        Args:
            x (float): Input value to evaluate.
        
        Returns:
            float: Membership degree (between 0 and 1).
        """
        return self._membership_function(x)
    
    @staticmethod
    def normalize_membership(membership_values):
        """
        Normalizes a set of membership values to ensure they stay within the [0,1] range.

        Args:
            membership_values (list or numpy array): A collection of membership values (float).

        Returns:
            numpy array: Normalized membership values, scaled between 0 and 1.
        """
        membership_values = np.array(membership_values)  # Convert input to a NumPy array
        
        # Find the maximum membership value (to scale)
        max_value = np.max(membership_values)
        
        # Prevent division by zero
        if max_value == 0:
            return membership_values  
        
        # Normalize by dividing all values by the maximum
        return membership_values / max_value
    
    def defuzzify(self, x_values, method="centroid"):
        """
        Defuzzifies the fuzzy term using the selected method.

        Args:
            accuracy(float): precision factor for plotting the graph. Multiplied by 100 to get the number of approximation points
            x_values (numpy array): Array of x-axis values corresponding to the fuzzy set.
            method (str): The defuzzification method to use.
                          Options: "centroid", "mom", "max", "weighted_avg".

        Returns:
            float: The defuzzified crisp value.
        """
        # Calculate membership values for each x in the given range
        membership_values = np.array([self.get_membership_value(x) for x in x_values])

        # Normalize the membership values
        membership_values = self.normalize_membership(membership_values)
        return defuzzify(x_values, membership_values, method)
        


class FuzzyLinguisticVariable:
    """
    Represents a fuzzy linguistic variable, which consists of multiple fuzzy terms and a universal range.

    Attributes:
        _name (str): The name of the linguistic variable.
        _universe (tuple): The range of possible values (min, max).
        _terms (dict): A dictionary mapping term names to their corresponding fuzzy terms.
    """
    def __init__(self, name: str, universe: tuple):
        self._name = name
        self._universe = universe  # (min, max) - The universal range of values
        self._terms = {}  # Terms with their membership functions

    def add_term(self, term: FuzzyTerm):
        """
        Adds a fuzzy term to the linguistic variable.

        Args:
            term (FuzzyTerm): The fuzzy term to be added.

        Raises:
            ValueError: If a term with the same name already exists.
        """
        if term.get_name() in self._terms:
            raise ValueError(f"Term '{term.get_name()}' already exists.")  
        self._terms[term.get_name()] = term

    def get_name(self):
        """Returns the name of the linguistic variable."""
        return self._name

    def get_universe(self):
        """Returns the universal range of values."""
        return self._universe

    def get_terms(self):
        """Returns a list of all term names within the linguistic variable."""
        return list(self._terms.keys())

    def get_membership(self, term_name: str, x):
        """
        Retrieves the membership value of a specific fuzzy term for the given input.

        Args:
            term_name (str): The name of the fuzzy term.
            x (float): The input value to evaluate.

        Raises:
            ValueError: If the specified term does not exist.

        Returns:
            float: The membership degree (between 0 and 1).
        """
        if term_name not in self._terms:
            raise ValueError(f"Term '{term_name}' not found.") 
        return self._terms[term_name].get_membership_value(x)

class FuzzyRule:
    """
    A fuzzy rule of the form:
    If Var1 is Term1 AND Var2 is Term2 ... then Y = f(...inputs...)

    Args:
        conditions (list): List of (linguistic_variable_name, term_name) pairs.
        consequence (function): Function that computes predicted output based on numeric inputs.
    """
    def __init__(self, conditions, consequence):
        self._conditions = conditions  # e.g., [("Volume", "Low"), ("RSI", "High")]
        self._consequence = consequence

    def evaluate_truth(self, inputs, variables_dict):
        """
        Computes the degree of truth of this rule for the given input values.

        Args:
            inputs (dict): {var_name: value}
            variables_dict (dict): {var_name: FuzzyLinguisticVariable}

        Returns:
            float: Minimum membership (truth degree) across all conditions.
        """
        degrees = []
        for var_name, term_name in self._conditions:
            var = variables_dict[var_name]
            x = inputs[var_name]
            degrees.append(var.get_membership(term_name, x))
        return min(degrees)

    def evaluate_output(self, inputs):
        """
        Evaluates the consequence function on raw numeric inputs.

        Args:
            inputs (dict): {var_name: value}

        Returns:
            float: Output Y.
        """
        return self._consequence(inputs)
class FuzzyRuleBase:
    """
    A base of fuzzy rules operating on multiple linguistic variables.

    Attributes:
        _rules (list): List of FuzzyRule instances.
        _variables (dict): {var_name: FuzzyLinguisticVariable}
    """
    def __init__(self, variables_dict):
        self._rules = []
        self._variables = variables_dict  # dictionary of variable_name -> FuzzyLinguisticVariable

    def add_rule(self, rule: FuzzyRule):
        self._rules.append(rule)

    def predict(self, input_dict):
        """
        Performs fuzzy inference to compute output based on current rules and input values.

        Args:
            input_dict (dict): {var_name: value}

        Returns:
            float: Final crisp output.
        """
        weighted_outputs = []
        weights = []

        for rule in self._rules:
            truth = rule.evaluate_truth(input_dict, self._variables)
            output = rule.evaluate_output(input_dict)
            weights.append(truth)
            weighted_outputs.append(truth * output)

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        return sum(weighted_outputs) / total_weight



"""

Main membership functions

"""

def triangle_function(a, b, c, inverse=False):
    """
    Creates a triangular membership function with an optional inverted version.

    Args:
        a (float): Left boundary of the triangle (membership 0).
        b (float): Peak of the triangle (membership 1).
        c (float): Right boundary of the triangle (membership 0).
        inverse (bool): If True, returns an **inverted** triangle.

    Returns:
        function: A function that calculates the membership value based on input x.
    """
    def mf(x):
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif a < x < b:
            value = (x - a) / (b - a)
        else:  # b < x < c
            value = (c - x) / (c - b)
        
        return 1.0 - value if inverse else value  # Invert if needed

    return mf


def trapezoidal_function(a, b, c, d, inverse=False):
    """
    Creates a trapezoidal membership function with an optional inverted version.

    Args:
        a (float): Left boundary (membership starts increasing).
        b (float): Start of plateau (membership = 1).
        c (float): End of plateau (membership = 1).
        d (float): Right boundary (membership decreases to 0).
        inverse (bool): If True, returns an **inverted** trapezoidal function.

    Returns:
        function: A function that calculates the membership value based on input x.
    """
    def mf(x):
        if x <= a or x >= d:
            return 0.0
        elif b <= x <= c:
            value = 1.0
        elif a < x < b:
            value = (x - a) / (b - a)
        else:  # c < x < d
            value = (d - x) / (d - c)

        return 1.0 - value if inverse else value  # Invert if needed

    return mf


def gaussian_function(mean, sigma, inverse=False):
    """
    Creates a Gaussian membership function with an optional inverted version.

    Args:
        mean (float): Peak center.
        sigma (float): Standard deviation (spread).
        inverse (bool): If True, returns an **inverted** Gaussian.

    Returns:
        function: A function that calculates the membership value based on input x.
    """
    def mf(x):
        value = math.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
        return 1.0 - value if inverse else value  # Invert if needed

    return mf


def sigmoid_function(a, b, inverse=False):
    """
    Creates a sigmoid membership function with an optional inverted version.

    Args:
        a (float): Steepness of curve.
        b (float): Midpoint.
        inverse (bool): If True, returns an **inverted** sigmoid.

    Returns:
        function: A function that calculates the membership value based on input x.
    """
    def mf(x):
        value = 1.0 / (1.0 + math.exp(min(700, -a * (x - b))))
        return 1.0 - value if inverse else value  # Invert if needed

    return mf


def z_shape_function(a, b, inverse=False):
    """
    Creates a Z-shaped membership function with an optional inverted version.

    Args:
        a (float): Start of transition.
        b (float): End of transition.
        inverse (bool): If True, returns an **S-shaped** function instead.

    Returns:
        function: A function that calculates the membership value based on input x.
    """
    def mf(x):
        if x <= a:
            value = 1.0
        elif x >= b:
            value = 0.0
        else:
            value = 1.0 - ((x - a) / (b - a)) ** 2
        
        return 1.0 - value if inverse else value  # Swap to S-shape if needed

    return mf

"""

Other support functions

"""


def plot_linguistic_variable(variable, accuracy = 1):
    """
    Plots the membership functions of all terms in a given fuzzy linguistic variable.

    Args:
        variable (FuzzyLinguisticVariable): The linguistic variable containing multiple fuzzy terms.
        accuracy(float): precision factor for plotting the graph. Multiplied by 100 to get the number of approximation points
    Returns:
        None: Displays a plot with the membership functions of all terms.
    """
    if accuracy <=0:
        accuracy = 1
    # Clear previous plots to prevent overlapping figures
    plt.clf()  

    # Retrieve universe range
    x_range = variable.get_universe()
    x_values = np.linspace(x_range[0], x_range[1], int(100 * accuracy))

    # Plot membership functions for each term in the linguistic variable
    for term_name in variable.get_terms():
        term = variable._terms[term_name]  # Get term object
        y_values = np.array([term.get_membership_value(x) for x in x_values])
        plt.plot(x_values, y_values, label=term_name)

    # Formatting the plot
    plt.xlabel("Input Value (x)")
    plt.ylabel("Membership Degree (μ)")
    plt.title(f"Membership Functions for '{variable.get_name()}'")
    plt.legend()
    plt.grid(True)
    plt.show()

def defuzzify(x_values, membership_values, method="centroid"):
    """
    Performs defuzzification using various techniques to convert fuzzy values into a crisp output.

    Args:
        x_values (numpy array): Array of x-axis values corresponding to the fuzzy set.
        membership_values (numpy array): Array of membership degrees for each x_value.
        method (str): The defuzzification method to use.
                      Options: "centroid", "mom", "max", "weighted_avg".

    Returns:
        float: The defuzzified crisp value.
    """
    # Ensure inputs are numpy arrays
    x_values = np.array(x_values)
    membership_values = np.array(membership_values)

    if method == "centroid":
        """
        **Centroid Method (Center of Gravity)**
        - Calculates the center of mass of the fuzzy membership function.
        - Formula: x* = Σ(x_i * μ(x_i)) / Σ(μ(x_i))
        - This method provides a balanced output based on all membership values.
        """
        numerator = np.sum(x_values * membership_values)
        denominator = np.sum(membership_values)
        return numerator / denominator if denominator != 0 else 0.0

    elif method == "mom":
        """
        **Mean of Maximum (MOM)**
        - Takes the average of all x-values where the membership is maximum.
        - Formula: x* = (x_max1 + x_max2 + ... + x_maxN) / N
        - Good for simple applications where multiple max values exist.
        """
        max_membership = np.max(membership_values)
        max_indices = np.where(membership_values == max_membership)[0]
        return np.mean(x_values[max_indices])

    elif method == "max":
        """
        **Maximum Membership Method**
        - Returns the x-value where membership is highest.
        - Formula: x* = x_max
        - Useful for fast decision-making processes where a single dominant output is needed.
        """
        max_index = np.argmax(membership_values)
        return x_values[max_index]

    elif method == "weighted_avg":
        """
        **Weighted Average Method**
        - Computes a weighted mean using membership values above a defined threshold.
        - Formula: x* = Σ(x_i * μ(x_i)) / Σ(μ(x_i))
        - Similar to centroid but considers only significant values.
        """
        threshold = np.max(membership_values) * 0.5  # Consider values above 50% of max membership
        filtered_indices = np.where(membership_values >= threshold)[0]
        numerator = np.sum(x_values[filtered_indices] * membership_values[filtered_indices])
        denominator = np.sum(membership_values[filtered_indices])
        return numerator / denominator if denominator != 0 else 0.0

    else:
        raise ValueError("Invalid defuzzification method. Choose from: 'centroid', 'mom', 'max', 'weighted_avg'.")

# Створення змінної
temperature = FuzzyLinguisticVariable("Температура", (0, 100))

# Додавання термів
temperature.add_term(FuzzyTerm("низька", triangle_function(0, 0, 50)))
temperature.add_term(FuzzyTerm("середня", triangle_function(25, 50, 75)))
temperature.add_term(FuzzyTerm("висока", triangle_function(50, 100, 100)))
#temperature.add_term(FuzzyTerm("t1", sigmoid_function(0.2, 10, True)))
#temperature.add_term(FuzzyTerm("t2", gaussian_function(2, 30, True)))
#temperature.add_term(FuzzyTerm("t3", z_shape_function(20, 50, True)))


# Отримання значення належності
x = 45
#print(f"Належність до 'середня' при {x}: {temperature.get_membership('середня', x)}")
print(f"Належність до 'низька' при {x}: {temperature.get_membership('низька', x)}")
#print(f"Належність до 'висока' при {x}: {temperature.get_membership('висока', x)}")
plot_linguistic_variable(temperature,3)


x_vals = np.array([0, 10, 20, 30, 40, 50])
membership_vals = np.array([0.1, 0.4, 0.8, 1.0, 0.8, 0.2])

# Example usage
print(defuzzify(x_vals, membership_vals, method="centroid"))  # Center of Gravity
print(defuzzify(x_vals, membership_vals, method="mom"))       # Mean of Maximum
print(defuzzify(x_vals, membership_vals, method="max"))       # Maximum Membership
print(defuzzify(x_vals, membership_vals, method="weighted_avg"))  # Weighted Average





# Створення змінних
volume = FuzzyLinguisticVariable("Volume", (0, 100))
volume.add_term(FuzzyTerm("Low", triangle_function(0, 0, 50)))
volume.add_term(FuzzyTerm("High", triangle_function(50, 100, 100)))

rsi = FuzzyLinguisticVariable("RSI", (0, 100))
rsi.add_term(FuzzyTerm("Low", triangle_function(0, 0, 50)))
rsi.add_term(FuzzyTerm("High", triangle_function(50, 100, 100)))

variables = {
    "Volume": volume,
    "RSI": rsi
}

# Створення правил
rule1 = FuzzyRule(
    conditions=[("Volume", "Low"), ("RSI", "High")],
    consequence=lambda inputs: 0.2 * inputs["Volume"] + 0.5 * inputs["RSI"]
)

rule2 = FuzzyRule(
    conditions=[("Volume", "High"), ("RSI", "Low")],
    consequence=lambda inputs: 0.4 * inputs["Volume"] - 0.3 * inputs["RSI"]
)

# Прогнозування
rule_base = FuzzyRuleBase(variables)
rule_base.add_rule(rule1)
rule_base.add_rule(rule2)

input_data = {
    "Volume": 30,
    "RSI": 70
}

print("Forecast:", rule_base.predict(input_data))
