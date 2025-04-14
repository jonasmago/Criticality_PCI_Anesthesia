"""
Edge-of-chaos measures
"""
# import dyst # for synthetic chaotic signal
import numpy as np
import scipy
import sklearn


def chaos_pipeline(data, sigma=0.5, denoise=False, downsample='minmax'):
    """Simplified pipeline for the modified 0-1 chaos test emulating the
    implementation from Toker et al. (2022, PNAS). This test assumes
    that the signal is stationarity and deterministic.

    Parameters
    ----------
    data : 1d array
        The (filtered) signal.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    denoise : bool
        If True, denoising will be applied according to the method by Schreiber
        (2000).
    downsample : str or bool
        If 'minmax', signal will be downsampled by conserving only local minima
        and maxima.

    Returns
    -------
    K: float
        Median K-statistic.

    """
    if denoise:
        # Denoise data using Schreiber denoising algorithm
        data = schreiber_denoise(data)

    if downsample == 'minmax':
        # Downsample data by preserving local minima
        data = _minmaxsig(data)

    # Check if signal is long enough, else return NaN
    if len(data) < 20:
        return np.nan

    # Normalize standard deviation of signal
    x = data * (0.5 / np.std(data)) # matlab equivalent  np.std(data,ddof=1)

    # Mdified 0-1 chaos test
    K = z1_chaos_test(x, sigma=sigma)

    return K


def z1_chaos_test(x, sigma=0.5, rand_seed=0):
    """Modified 0-1 chaos test. For long time series, the resulting K-statistic
    converges to 0 for regular/periodic signals and to 1 for chaotic signals.
    For finite signals, the K-statistic estimates the degree of chaos.

    Parameters
    ----------
    x : 1d array
        The time series.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    rand_seed : int, optional
        Seed for random number generator. The default is 0.

    Returns
    -------
    median_K : float
        Indicator of chaoticity. 0 is regular/stable, 1 is chaotic and values
        in between estimate the degree of chaoticity.

    References
    ----------
    Gottwald & Melbourne (2004) P Roy Soc A - Math Phy 460(2042), 603-11.
    Gottwald & Melbourne (2009) SIAM J Applied Dyn Sys 8(1), 129-45.
    Toker et al. (2022) PNAS 119(7), e2024455119.
    """
    np.random.seed(rand_seed)
    N = len(x)
    j = np.arange(1,N+1)
    t = np.arange(1,int(round(N / 10))+1)
    M = np.zeros(int(round(N / 10)))
    # Choose a coefficient c within the interval pi/5 to 3pi/5 to avoid
    # resonances. Do this 1000 times.
    c = np.pi / 5 + np.random.random_sample(1000) * 3 * np.pi / 5
    k_corr = np.zeros(1000)

    for its in range(1000):
        # Create a 2-d system driven by the data
        #p = cumsum(x * cos(a * c[i]))
        #q = cumsum(x * sin(a * c[i]))
        p=np.cumsum(x * np.cos(j*c[its]))
        q=np.cumsum(x * np.sin(j*c[its]))

        for n in t:
            # Calculate the (time-averaged) mean-square displacement,
            # subtracting a correction term (Gottwald & Melbourne, 2009)
            # and adding a noise term scaled by sigma (Dawes & Freeland, 2008)

            #M[n-1]=(np.mean((p[n+1:N] - p[1:N-n])**2 + (q[n+1:N]-q[1:N-n])**2)
            #      - np.mean(x)**2 * (1-np.cos(n*c[its])) / (1-np.cos(c[its]))
            #      + sigma * (np.random.random()-.5))

            M[n-1]=(np.mean((p[n:N] - p[:N-n])**2 + (q[n:N]-q[:N-n])**2)
                  - np.mean(x)**2 * (1-np.cos(n*c[its])) / (1-np.cos(c[its]))
                  + sigma * (np.random.random()-.5))

        k_corr[its], _ = scipy.stats.pearsonr(t, M)
        median_k = np.median(k_corr)

    return median_k

def _minmaxsig(x):
    maxs = scipy.signal.argrelextrema(x, np.greater)[0]
    mins = scipy.signal.argrelextrema(x, np.less)[0]
    minmax = np.concatenate((mins, maxs))
    minmax.sort(kind='mergesort')
    return x[minmax]
