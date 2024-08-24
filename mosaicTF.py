import numpy as np
import soundfile as sf
from scipy.signal import stft
import matplotlib.pyplot as plt


def mosaicTF(signalAVGs, nperseg=512):
    '''
    Performs MosaicTF computation
    Ref.: 2024 Prawda - Non-stationary Noise Removal from Repeated Sweep Measurements

    Parameters
    ----------
    signalAVGs : np.array  [samples x avgs]
        Array containing sweeps data with all the averages in the last dimension
    nperseg: int, optional
        Length of each STFT segment. Defaults to 256.
    '''
    STFTall = []
    for ch in range(signalAVGs.shape[-1]):
        f, t, Zxx = stft(signalAVGs[..., ch], fs, nperseg=nperseg)
        STFTall.append(Zxx)

    STFTall = np.array(STFTall)

    processed = (np.median(np.real(STFTall), axis=0) +
                 1j*np.median(np.imag(STFTall), axis=0))
    processed = np.squeeze(processed)
    return processed, f, t





# Load sweep averages
data, fs = sf.read(
    r'Case - measured sweeps - Fig.2\3_sweeps_speech_transient.wav')
nSweeps = 3
sweepLenSmaples = data.shape[0] // nSweeps

# Split sweep avgs
sweepAvgs = []
for ch in range(nSweeps):
    sweepAvgs.append(data[ch*sweepLenSmaples: sweepLenSmaples*(ch+1)])
sweepAvgs = np.array(sweepAvgs).T


# process Mosaic-TF -----------------------------------
nperseg = 512
processed, f_proc, t_proc = mosaicTF(sweepAvgs, nperseg=nperseg)



# Plot  =============================================================================
vmin = -120
vmax = -50
f, t, Zxx = stft(data, fs, nperseg=nperseg)
plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)),
               vmin=vmin, vmax=vmax, shading='gouraud')
plt.title('Raw Input')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


plt.pcolormesh(t_proc, f_proc, 20*np.log10(np.abs(processed)),
               vmin=vmin, vmax=vmax, shading='gouraud')
plt.title('Mosaic-TF')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


data, fs = sf.read(
    r'Case - measured sweeps - Fig.2\3_sweeps_speech_transient_med_filt_tf.wav')
f, t, Zxx = stft(data, fs, nperseg=nperseg)
plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx)),
               shading='gouraud',  vmin=vmin, vmax=vmax)
plt.title('Reference Mosaic-TF')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
