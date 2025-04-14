import pandas as pd
import numpy as np
import scipy
import neurokit2 as nk
from numpy import (absolute, angle, arctan2,
                   diff, exp, imag, mean, real, square,
                   var)
from scipy.signal import hilbert
from scipy.io import loadmat

def pcf(data):
    """Estimate the pair correlation function (PCF) in a network of
    oscillators, equivalent to the susceptibility in statistical physics.
    The PCF shows a sharp peak at a critical value of the coupling between
    oscillators, signaling the emergence of long-range correlations between
    oscillators.

    Parameters
    ----------
    data : 2d array
        The filtered input data, where the first dimension is the different
        oscillators, and the second dimension is time.

    Returns
    -------
    pcf : float
        The pair correlation function, a scalar value >= 0.
    orpa: float
        Absolute value of the order parameter (degree of synchronicity)
        averaged over time, being equal to 0 when the oscillators’ phases are
        uniformly distributed in [0,2π ) and 1 when they all have the same
        phase.
    orph_vector: 1d array (length = N_timepoints)
        Order parameter phase for every moment in time.
    orpa_vector: 1d array (length = N_timepoints)
        Absolute value of the order parameter (degree of synchronicity) for
        every moment in time.

    References
    ----------
    Yoon et al. (2015) Phys Rev E 91(3), 032814.
    """

    N_ch = min(data.shape)  # Nr of channels
    N_time = max(data.shape)  # Nr of channels

    # inifialize empty array
    inst_phase = np.zeros(data.shape)
    z_vector = []

    # calculate Phase of
    for i in range(N_ch):
        inst_phase[i,:] = np.angle(scipy.signal.hilbert(data[i,:]))

    for t in range(N_time):
        # get global synchronization order parameter z over time
        z_vector.append(np.mean(exp(1j * inst_phase[:,t])))

    z_vector = np.array(z_vector)

    #  r =|z| degree of synchronicity
    orpa_vector = abs(z_vector)
    # get order phases
    orph_vector = arctan2(imag(z_vector), real(z_vector))

    # get PCF = variance of real part of order parameter
    # var(real(x)) == (mean(square(real(x))) - square(mean(real(x))))
    pcf = N_ch * var(real(z_vector))
    # time-averaged Order Parameter
    orpa = mean(orpa_vector);

    return pcf, orpa, orph_vector, orpa_vector

