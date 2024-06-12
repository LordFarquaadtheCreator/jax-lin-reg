import jax.numpy as jnp
from jax import grad, jit
from get_slope import get_slope


def get_intercept(x, y, b):
	'''a = y' - bx' linear regression'''
	mean_x: float = jnp.mean(x)
	mean_y: float = jnp.mean(y)
	a: float = mean_y - (b * mean_x)
    
	return a