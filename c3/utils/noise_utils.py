"""Useful functions to read, fit and interchanging between noise representations."""
import numpy as np
import random
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft, irfft, rfft


def load_spectrum(filename, delimiter="\t"):
    tmp = np.loadtxt(filename, delimiter=delimiter)
    # freq = tmp[:,0]
    # dens = tmp[:,1]
    return tmp


def calc_auto_correlation(dens):
    return np.fft.fft(dens)


def calc_spectrum(auto_corr):
    return np.fft.ifft(auto_corr)


def lorentz(w, g, lam):
    return g ** 2 * 4 * lam / (4 * lam ** 2 + w ** 2)


def oneoverf(A, alpha, w):
    return A * 1 / np.power(w, alpha)


def sum_of_tls(w, *params):
    y = np.zeros_like(w)
    for i in range(0, len(params), 2):
        g = params[i]
        lam = params[i + 1]
        y = y + g ** 2 * 4 * lam / (4 * lam ** 2 + w ** 2)
    return y


def fit_spectrum_with_tls(spec, tls_number=5, fevals=10000, guess=None):
    xdata = spec[:, 0]
    ydata = spec[:, 1]
    if guess == None:
        guess = []
        for i in range(tls_number):
            guess.append(1e-3)
            guess.append(1e-5)
    popt, pcov = curve_fit(sum_of_tls, xdata, ydata, guess, maxfev=fevals)
    return popt, pcov


def generate_time_shot(spec, stepsize):
    fmax = max(spec[:, 0])
    xvals = np.linspace(spec[0, 0], fmax, stepsize)
    dens = np.interp(xvals, spec[:, 0], spec[:, 1])
    phase = 2 * np.pi * np.random.uniform(0, 1, stepsize)
    dt = 1e-10  # 1/fmax
    times = np.arange(0, (stepsize - 1) * dt, dt / 2)
    return xvals, times, np.fft.irfft(dens * np.exp(1j * phase))

    # wplus=np.arange(min(freqs),max(freqs),1./step_number*(max(freqs)-min(freqs)))
    # wmin=-np.flip(wplus)
    # w=np.concatenate((wmin,wplus),axis=0)
    # psdplus=np.interp(wplus,freqs,dens)
    # psdmin=np.flip(psdplus)
    # psd=np.concatenate((psdmin,psdplus),axis=0)
    # Amplitude=np.sqrt(np.abs(psd))+0j
    # phase=[]
    # for i in range(step_number):
    #    phase.append(random.uniform(0,2*np.pi))
    # phase=np.concatenate((-np.flip(phase),phase),axis=0)
    # phase=np.exp(1j*phase)
    # Z=np.sqrt(2)*Amplitude*phase
    # return np.real(ifft(Z))


##testing
# real=np.random.normal(0,1,20)
# spec=load_spectrum('data_parity.dat')
# freq=spec[:,0]
# dens=spec[:,1]
# popt,pcov=fit_spectrum_with_tls(spec,1)
#
# fig=plt.figure()
# plt.plot(freq,dens,'.r')
# plt.plot(freq,sum_of_tls(freq,*popt))
# plt.show()
# res=calc_time_shot(spec,2**10+1)
# simspec=0*res
# freq2plus=np.arange(min(freq),max(freq),1./1025*(max(freq)-min(freq)))
# freq2min=-np.flip(freq2plus)
# freq2=np.concatenate((freq2min,freq2plus),axis=0)
# dens2plus=np.interp(freq2plus,freq,dens)
# dens2min=np.flip(dens2plus)
# dens2=np.concatenate((dens2min,dens2plus),axis=0)
# for i in range(0,1025):
#    res=calc_time_shot(spec,2**10+1)
#    erg=np.abs(fft(res))**2
#    simspec+=erg
# for a in res:
#    print(a)
# simspec=simspec/1025
#
# plt.plot(freq2,simspec,'.r',label="simulated")
# plt.plot(freq2,dens2)
# plt.show()
