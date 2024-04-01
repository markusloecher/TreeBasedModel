import numpy as np
import numbers

def simulate_data_strobl(n=120, # number of rows in data
                         relevance=0.15, # signal strength (0 for NULL)
                         seed=None):
    """Simulate Strobl-like dataset with corrected dimensions."""

    if isinstance(seed, numbers.Integral) or (seed is None):
        random_state = np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        random_state = seed

    # Assuming you want different types of features but all flattened to a 1D array per type
    x1 = random_state.standard_normal(size=n)
    x2 = random_state.randint(1, 3, size=n)
    x3 = random_state.randint(1, 5, size=n)
    x4 = random_state.randint(1, 11, size=n)
    x5 = random_state.randint(1, 21, size=n)
    # Stack them to get a shape of (n, 5) - assuming you want each x as a feature
    X = np.column_stack([x1, x2, x3, x4, x5])

    # Adjust the target generation as per your logic
    # Example assuming relevance impacts the probability linearly for simplicity
    y = random_state.binomial(n=1, p=0.5 + relevance * (x2 - 1) * 2.0)

    return X, y