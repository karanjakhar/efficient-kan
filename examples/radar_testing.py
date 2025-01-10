import sys
sys.path.append('../src')

from efficient_kan import KAN


# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm



import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Peak Interpolation Functions
def peak_interp_3pts_gaussian(yleft, ymid, yright):
    """
    Gaussian interpolation using 3 points.

    Args:
        yleft (float): Left point value.
        ymid (float): Middle point value.
        yright (float): Right point value.

    Returns:
        tuple: A tuple containing the interpolated peak height and delta value.
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
snr_linear = 1             # Linear SNR value

np.random.seed(15)            # Set random seed for reproducibility

# Monte Carlo Simulation
fbeat_stack = np.zeros(MC_iters)
f_est2 = np.zeros(MC_iters)
err_f_est2 = np.zeros(MC_iters)
err_R_est2 = np.zeros(MC_iters)

all_sig = []
for mc_i in range(MC_iters):
    # Generate random beat frequency within the specified range
    fbeat = 1e6 * (fmin_MHz + (fmax_MHz - fmin_MHz) * np.random.rand())
    fbeat_stack[mc_i] = fbeat

    # Generate signal with the beat frequency and additive noise
    t = np.arange(N) / fs
    sig = np.cos(2 * np.pi * fbeat * t) + np.sqrt(0.5 / snr_linear) * np.random.randn(N)
    all_sig.append(sig)

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
    

label = (fbeat_stack * Tc * c / (2 * BW))/R_axis[-1]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# train_input = all_sig[:8000]
# train_label = label[:8000]
val_input = all_sig
val_label = label

# Load MNIST
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
# )
# trainset = torchvision.datasets.MNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# valset = torchvision.datasets.MNIST(
#     root="./data", train=False, download=True, transform=transform
# )
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
# valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define model
# model = KAN([4096, 64, 10])
model = torch.load("model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # Define optimizer
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# # Define learning rate scheduler
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)


# Print the trainable parameters
# print("Trainable Parameters:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()} elements")

# # Optional: print the total number of trainable parameters
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"\nTotal Trainable Parameters: {total_params}")
# Define loss
# criterion = nn.CrossEntropyLoss()

criterion = nn.MSELoss()


model.eval()
val_loss = 0
val_accuracy = 0
all_losses = []
all_error_m_ranges = []
with torch.no_grad():
    for input, label in zip(val_input, val_label):
        input = torch.tensor(np.array(input), dtype=torch.float32).to(device)
        output = model(input)
        # print(output.size())
        error_m_range = np.array(label) - output.cpu().numpy()
        loss = torch.sqrt(criterion(output, torch.tensor(np.array(label), dtype=torch.float32).to(device))).item()
        all_losses.append(loss)
        all_error_m_ranges.append(error_m_range.tolist()[0])
        val_loss += loss
        
val_loss /= len(val_input)



print(
    f"KAN RMSE Loss: {val_loss}"
)



# Calculate RMSE for the interpolation method
err_R_est_RMSE2 = np.sqrt(np.mean(err_R_est2**2))

print("Interpolation method RMSE loss:", err_R_est_RMSE2)

e = np.array(all_error_m_ranges)
# Fit Probability Distributions
pd_Gaussian = gaussian_kde(e)

# Calculate and Plot Probability Density Functions
CRB = fs**2 * 6 / (snr_linear * (N/2) * ((N/2)**2 - 1)) / 4 / np.pi**2

x = np.linspace(-0.08, 0.08, 100)
y_Gaussian = pd_Gaussian(x)
y_CRB = 1 / np.sqrt(2 * np.pi * (Tc * c / (2 * BW))**2 * CRB) * np.exp(-x**2 / (2 * (Tc * c / (2 * BW))**2 * CRB))

print(len(y_Gaussian), len(y_CRB))

plt.figure(figsize=(8, 6))
plt.plot(x, y_Gaussian, 'r-s', linewidth=2, label='Gaussian')
plt.plot(x, y_CRB, 'k--', linewidth=2, label='CRLB')
plt.xlabel('Range Error (m)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# print(all_error_m_ranges)

sorted_error_m_ranges = sorted(all_error_m_ranges)
plt.figure(figsize=(8, 6))
plt.plot(sorted_error_m_ranges, list(range(1,10001)), 'r-s', linewidth=2, label='KAN')
plt.xlabel('Range Error (m)')
plt.ylabel('Iterations')
plt.legend()
plt.grid(True)
plt.show()