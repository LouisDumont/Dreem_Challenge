from scipy.signal import welch

def own_welch(signal, nperseg=128):
    #print(welch(signal, fs=250, nperseg=nperseg)[0])
    return welch(signal, fs=250, nperseg=nperseg)[1]