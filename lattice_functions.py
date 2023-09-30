import numpy as np

from numba import njit


@njit(cache=True)
def new_configuration(temperature: float,
                      sigma: np.ndarray,
                      steps: int
                      ) -> np.ndarray:
    """
    This function models the evolution of the Ising model after a given
    amount of steps.

    Parameters
    ----------
    temperature : float
        The temperature of the material.
    sigma : ndarray
        The lattice of electrons.
    steps : int
        The number of iterations.

    Returns
    -------
    sigma : ndarray
        The lattice after `steps` amount of iterations.

    Examples
    --------
    >>> import numpy as np
    >>> sigma = np.ones((20, 20))
    >>> new_sigma = new_configuration(2., sigma, 3)
    """
    shape = sigma.shape

    sigma = sigma.copy()

    for _ in range(steps):
        # Randomly select an electron on the lattice using these numbers
        i = np.random.randint(0, shape[0])
        j = np.random.randint(0, shape[1])

        # Change in lattice energy with electron at (i, j) flipped
        delta_energy = 2*sigma[i, j]*(sigma[i - 1, j] + sigma[i, j - 1] +
                                      sigma[(i + 1) % shape[0], j] +
                                      sigma[i, (j + 1) % shape[1]])

        # Decide if lattice change is kept
        if not (delta_energy > 0 and np.exp(-delta_energy / temperature) < np.random.random()):
            sigma[i, j] *= -1
        
    return sigma
