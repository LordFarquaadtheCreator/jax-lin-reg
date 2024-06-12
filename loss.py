def mse(y_acc: list, y_pred: list):
    import jax.numpy as np
    # MSE = (1/n) * Σ(yᵢ - ȳ)²
