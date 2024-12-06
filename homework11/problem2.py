from scipy.signal import lfilter
import matplotlib.pyplot as plt

# Apply the filter
ecg_filtered = lfilter(b, a, ecg_noisy)

# Plot the original and filtered signals
fs = 300  # Sampling frequency
time = np.linspace(0, len(ecg_noisy) / fs, len(ecg_noisy))

plt.figure(figsize=(12, 6))
plt.plot(time[:600], ecg_noisy[:600], label="Noisy ECG", alpha=0.5)
plt.plot(time[:600], ecg_filtered[:600], label="Filtered ECG", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Filtered ECG Signal")
plt.legend()
plt.grid()
plt.show()
### Copy and paste the code from your designed filter function here in this file. ###
