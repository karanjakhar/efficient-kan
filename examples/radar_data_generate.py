"""Copyright © 2024 alephzero.ai by Dr. Nir RegevAll rights reserved. 
This code or any portion thereofmay not be reproduced or used in any manner whatsoeverwithout 
the express written permission of the author except for the useof brief quotations in a review 
or scholarly presentation.This code is provided "as is", without warranty of any kind, express 
orimplied, including but not limited to the warranties of merchantability,fitness for a particular 
purpose and noninfringement. In no event shall theauthor be liable for any claim, damages or other 
liability, whether in anaction of contract, tort or otherwise, arising from, out of or inconnection 
with the software or the use or other dealings in the software.For usage permission requests, write 
to the author, addressed "Attention: Permissions Coordinator,"at the address below:
Dr. Nir Regevnir@alephzero.ai
Created: 04/19/2024Last 
Updated: 05/07/2024
Description:
This Python script is developed for educational and research purposes in the field of signal processing.
It demonstrates advanced radar signal processing techniques such as Monte Carlo simulations and peak 
interpolationmethods to estimate frequency and range errors. The script utilizes various NumPy functions 
and customalgorithms to process and analyze simulated radar signals.
"""

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Peak Interpolation Functions
def peak_interp_3pts_gaussian(yleft, ymid, yright):    
    """    Gaussian interpolation using 3 points.    
    Args:        
    yleft (float): Left point value.        
    ymid (float): Middle point value.        
    yright (float): Right point value.    
    Returns:        tuple: A tuple containing the interpolated peak height and delta value.    
    """    
    alpha = yleft    
    beta = ymid    
    gamma = yright    
    delta = 0.5 * (np.log(alpha) - np.log(gamma)) / (np.log(alpha) - 2 * np.log(beta) + np.log(gamma))    
    peak_height = np.exp(np.log(beta) - 0.25 * (np.log(alpha) - np.log(gamma)) * delta)    
    return peak_height, delta 

# Constants
fs = 100e6                    # Sampling frequency (Hz)
c = const.speed_of_light      # Speed of light (m/s)
Ts = 1 / fs                   # Sampling period (s)
BW = 1e9                      # Bandwidth (Hz)
N = 4096                      # Number of samples
Tc = N / fs                   # Chirp duration (s)
# Frequency and Range Axes
df = fs / N                   # Frequency resolution (Hz)
f_axis = np.arange(0, fs/2, df)   # Frequency axis (Hz)
R_axis = Tc * c / (2 * BW) * f_axis   # Range axis (m)
# Monte Carlo Simulation Parameters
MC_iters = int(10e3)          # Number of Monte Carlo iterations
fmin_MHz = 5                  # Minimum beat frequency (MHz)
fmax_MHz = 10                 # Maximum beat frequency (MHz)
max_val = np.zeros(MC_iters)  # Array to store maximum values
snr_linear = 0.1              # Linear SNR value
np.random.seed(15)            # Set random seed for reproducibility

# Monte Carlo Simulation
fbeat_stack = np.zeros(MC_iters)
f_est2 = np.zeros(MC_iters)
err_f_est2 = np.zeros(MC_iters)
err_R_est2 = np.zeros(MC_iters)
for mc_i in range(MC_iters):    
    # Generate random beat frequency within the specified range    
    fbeat = 1e6 * (fmin_MHz + (fmax_MHz - fmin_MHz) * np.random.rand())    
    fbeat_stack[mc_i] = fbeat    
    
    # Generate signal with the beat frequency and additive noise    
    t = np.arange(N) / fs    
    sig = np.cos(2 * np.pi * fbeat * t) + np.sqrt(0.5 / snr_linear) * np.random.randn(N)   
    # Apply Hanning window and compute the FFT    
    SIG1 = (1 / np.sqrt(N)) * np.fft.fft(sig * np.hanning(len(sig)), N)    
    SIG = SIG1[:N//2]    
    # Find the maximum peak in the spectrum    
    max_val[mc_i] = np.max(np.abs(SIG)**2)    
    ind_max = np.argmax(np.abs(SIG)**2)    
    # Interpolation    
    sl = np.abs(SIG[ind_max - 1])**2  # Left point    
    sm = np.abs(SIG[ind_max])**2      # Middle point    
    sr = np.abs(SIG[ind_max + 1])**2  # Right point    
    if sm > sl and sm > sr:        
        # Perform peak interpolation using Gaussian interpolation        
        yp2, delta2 = peak_interp_3pts_gaussian(sl, sm, sr)        
        
        # Estimate frequencies and ranges using the interpolated peak locations        
        f_est2[mc_i] = f_axis[ind_max] + delta2 * df        
        
        # Calculate frequency and range errors for the interpolation method        
        err_f_est2[mc_i] = f_est2[mc_i] - fbeat        
        err_R_est2[mc_i] = err_f_est2[mc_i] * Tc * c / (2 * BW)    
        
    else:        
        raise ValueError('Peak detection failed.')
    

# Calculate RMSE for the interpolation method
err_R_est_RMSE2 = np.sqrt(np.mean(err_R_est2**2))

# Fit Probability Distributions
pd_Gaussian = gaussian_kde(err_R_est2)

# Calculate and Plot Probability Density Functions

CRB = fs**2 * 6 / (snr_linear * (N/2) * ((N/2)**2 - 1)) / 4 / np.pi**2

x = np.linspace(-0.08, 0.08, 100)
y_Gaussian = pd_Gaussian(x)
y_CRB = 1 / np.sqrt(2 * np.pi * (Tc * c / (2 * BW))**2 * CRB) * np.exp(-x**2 / (2 * (Tc * c / (2 * BW))**2 * CRB))

plt.figure(figsize=(8, 6))
plt.plot(x, y_Gaussian, 'r-s', linewidth=2, label='Gaussian')
plt.plot(x, y_CRB, 'k--', linewidth=2, label='CRLB')
plt.xlabel('Range Error (m)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()