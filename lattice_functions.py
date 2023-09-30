import numpy as np

from numba import njit


def lattice_energy(sigma: np.ndarray) -> int:
    """
    Calculate the lattice energy without any external magnetic
    field.

    Parameters
    ----------
    sigma : ndarray
        The lattice of electron spins.

    Returns
    -------
    energy : int
        The energy of the lattice.

    Notes
    -----
    For a lattice of dipoles that are not influenced by an external
    magnetic field, the Hamiltonian is the sum of the product of the
    spins of adjacent electrons multiplied by negative one.

    .. math::
        H = -\sum_{\langle i,j \in \left| i - j \right| = 1 \rangle} \sigma_i\sigma_j

    Examples
    --------
    >>> import numpy as np
    >>> sigma = np.array([[1, 1, 1], [-1, 1, 1]])
    >>> lattice_energy(sigma)
    -4
    """
    energy = -int(np.sum(sigma*np.roll(sigma, 1, axis=0)) + np.sum(sigma*np.roll(sigma, 1, axis=1)))
    return energy


@njit(cache=True)
def new_configuration(temperature: float,
                      sigma: np.ndarray,
                      energy: int,
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
    energy : int
        The energy of the lattice.
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
    >>> new_sigma = new_configuration(2., sigma, lattice_energy(sigma), 3)
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
        if not (delta_energy > 0 and np.exp(-delta_energy / temperature) >= np.random.random()):
            sigma[i, j] = -sigma[i, j]
            energy += delta_energy
        
    return sigma
