import sys
import numpy as np
import os
import matplotlib.pyplot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime
from tqdm import tqdm
import GPy
from collections import defaultdict
from pathlib import Path
import seaborn as sns
import scipy.stats as stats
from matplotlib.colors import ListedColormap
import warnings
import time
from itertools import product
from joblib import Parallel, delayed
from operator import itemgetter
import sncosmo
import emcee
import corner
import pickle
from multiprocessing import Pool
from Emulator.Classes.GP5D import GP5D

mpl.rcParams['legend.frameon'] = False
mpl.rcParams['figure.autolayout'] = True
# mpl.rcParams['figure.dpi'] = 300
# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = False


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


def utkarshGrid():
    plt.minorticks_on()
    plt.grid(color='grey',
             which='minor',
             linestyle=":",
             linewidth='0.1',
             )
    plt.grid(color='black',
             which='major',
             linestyle=":",
             linewidth='0.1',
             )


class Inference():
    def __init__(self):
        self.truth_arr = np.array([None, None, None, None])
        self.curr_wv = np.arange(100, 3600, 10)
        self.set_skip_factor = None
        self.gp = GP5D("Classes/reference.csv")
        self.gp.MCMC = True
        self.gp.split = 1
        self.gp.emulator = "start"
        self.mejdyn_guess, self.mejwind_guess, self.phi_guess, self.iobs_guess = [0.1, 0.1, 20, 9]
        self.theta = None
        self.pool = None
        self.nwalkers = 10
        self.nburn = 2
        self.niter = 20
        self.state = None
        self.samples = None
        self.labs = None
        pass

    def train_fluxes(self, messages=False):
        t0 = time.time()
        mejdyn, mejwind, phi, iobs = list(self.truth_arr)
        curr_wv = self.curr_wv
        set_skip_factor = self.set_skip_factor
        gp = self.gp
        gp.cross_validation = (mejdyn, mejwind, phi, iobs)
        gp.set_wv_range(curr_wv)
        gp.n_comp = 25
        gp.save_pca_components(skip_factor=set_skip_factor)
        gp.setXY_cross_validation(mejdyn, mejwind, phi, iobs, messages=False)
        fitting_kernel = GPy.kern.RBF(input_dim=4, variance=1, lengthscale=1, ARD=True)
        decay_kernel = GPy.kern.Linear(input_dim=4, ARD=True)
        gp.kernel = fitting_kernel * decay_kernel
        gp.model = GPy.models.GPRegression(gp.X, gp.Y, gp.kernel)
        if messages:
            print(gp.model)
        if messages:
            print(f"[STATUS] Optimizing...")
        gp.model.optimize(messages=False)
        gp.model_predict(include_like=True, messages=False)
        gp.save_pca_initial_validation()
        gp.delete_folder_files("data/pcaComponentsTrained")
        gp.delete_folder_files("data/pcaComponentsTrainedError")
        self.gp = gp
        print(f"Training Time: {round(time.time() - t0)}s")
        return None

    def normalize_mejdyn(self, x):
        max = self.gp.reference.mejdyn.max()
        min = self.gp.reference.mejdyn.min()
        num = x - min
        dem = max - min
        normalized = num / dem
        return normalized

    def undo_normalize_mejdyn(self, x_prime):
        max = self.gp.reference.mejdyn.max()
        min = self.gp.reference.mejdyn.min()
        unnormalized = min + x_prime * (max - min)
        return unnormalized

    def normalize_mejwind(self, x):
        max = self.gp.reference.mejwind.max()
        min = self.gp.reference.mejwind.min()
        num = x - min
        dem = max - min
        normalized = num / dem
        return normalized

    def undo_normalize_mejwind(self, x_prime):
        max = self.gp.reference.mejwind.max()
        min = self.gp.reference.mejwind.min()
        unnormalized = min + x_prime * (max - min)
        return unnormalized

    def normalize_phi(self, x):
        max = self.gp.reference.phi.max()
        min = self.gp.reference.phi.min()
        num = x - min
        dem = max - min
        normalized = num / dem
        return normalized

    def undo_normalize_phi(self, x_prime):
        max = self.gp.reference.phi.max()
        min = self.gp.reference.phi.min()
        unnormalized = min + x_prime * (max - min)
        return unnormalized

    def normalize_iobs(self, x):
        max = 10
        min = 0
        num = x - min
        dem = max - min
        normalized = num / dem
        return normalized

    def undo_normalize_iobs(self, x_prime):
        max = 11
        min = 0
        unnormalized = min + x_prime * (max - min)
        return unnormalized

    def normalization_helper(self, X):
        normed = np.zeros(X.shape)
        normed[0] = self.normalize_mejdyn(X[0])
        normed[1] = self.normalize_mejwind(X[1])
        normed[2] = self.normalize_phi(X[2])
        normed[3] = self.normalize_iobs(X[3])
        return np.array(normed, dtype=float)

    def undo_normalization_helper(self, X_prime):
        unnormed = np.zeros(X_prime.shape)
        unnormed[0] = self.undo_normalize_mejdyn(X_prime[0])
        unnormed[1] = self.undo_normalize_mejwind(X_prime[1])
        unnormed[2] = self.undo_normalize_phi(X_prime[2])
        unnormed[3] = self.undo_normalize_iobs(X_prime[3])
        return unnormed

    def undo_normalization_samples(self, samples):
        a = np.copy(samples)
        a[:, :, 0] = self.undo_normalize_mejdyn(a[:, :, 0])
        a[:, :, 1] = self.undo_normalize_mejwind(a[:, :, 1])
        a[:, :, 2] = self.undo_normalize_phi(a[:, :, 2])
        a[:, :, 3] = self.undo_normalize_iobs(a[:, :, 3])
        return a

    def predict_fluxes(self, mejdyn, mejwind, phi, iobs, extra_item=None):
        theta = np.array([mejdyn, mejwind, phi, iobs])
        mejdyn, mejwind, phi, iobs = list(self.undo_normalization_helper(theta))
        gp = self.gp
        gp.validationX = [mejdyn, mejwind, phi, iobs]
        gp.validationXNormed = self.normalization_helper(np.array(gp.validationX))
        gp.model_predict_cross_validation(include_like=True, messages=False)  # Save cross validation
        gp.save_trained_data(errors=False, theta=(mejdyn, mejwind, phi, iobs), extra_item=extra_item)
        y = np.load(f"data/pcaTrained/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        os.remove(f"data/pcaComponentsTrained/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        os.remove(f"data/pcaComponentsTrainedError/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        filters = ["sdss::u", "sdss::g", "sdss::r", "sdss::i", "sdss::z",
                   "swope2::y", "swope2::J", "swope2::H"]

        m_complete = []
        t = gp._t_helper()
        for i in range(len(filters)):
            source = sncosmo.TimeSeriesSource(t, gp.wv_range * 10, 10 ** y)
            m = source.bandmag(filters[i], "ab", t)
            m_complete.append(m)

        y_mag = np.array(m_complete, dtype=float).T
        t_matrix = np.repeat(t, len(filters)).reshape(len(t), len(filters))
        x = t_matrix
        self.filters = filters
        return x, y_mag

    def main(self, yerr_percentage=20):
        self.truth_arr_normed = self.normalization_helper(self.truth_arr)
        x, y = self.predict_fluxes(mejdyn=self.truth_arr_normed[0],
                                   mejwind=self.truth_arr_normed[1],
                                   phi=self.truth_arr_normed[2],
                                   iobs=self.truth_arr_normed[3],
                                   extra_item=True)
        print(y.shape)
        yerr = yerr_percentage / 100 * y
        self.initial = np.array([self.mejdyn_guess, self.mejwind_guess, self.phi_guess, self.iobs_guess])
        self.initial_normalized = self.normalization_helper(self.initial)

        def prior(theta):
            mejdyn, mejwind, phi, iobs = theta
            if mejdyn > 1.01 or mejdyn < -0.01:
                return -np.inf
            elif mejwind > 1.01 or mejwind < -0.01:
                return -np.inf
            elif phi > 1.01 or phi < -0.01:
                return -np.inf
            elif iobs > 1.01 or iobs < -0.01:
                return -np.inf
            else:
                return 0.0

        def loglike(theta, x, y, yerr):
            mejdyn, mejwind, phi, iobs = theta
            x, y_model = self.predict_fluxes(mejdyn, mejwind, phi, iobs)
            logl = - 0.5 * np.sum(((y - y_model) / yerr) ** 2)
            return logl

        def logpost(theta, x=x, y=y, yerr=yerr):
            # print(theta, self.undo_normalization_helper(theta))
            if not np.isfinite(prior(theta)):
                return -np.inf
            logp = prior(theta)  # prior
            logl = loglike(theta, x=x, y=y, yerr=yerr)  # likelihood
            return logl + logp  # posterior

        initial = self.initial_normalized
        ndim = len(initial)
        self.ndim = ndim
        self.t_init = time.time()
        p0 = [np.array(initial) + 1e-4 * np.random.randn(ndim) for i in range(self.nwalkers)]
        p0 = np.array(p0, dtype=float)

        with Pool() as pool:
            pool = None
            sampler = emcee.EnsembleSampler(self.nwalkers, ndim, logpost, pool=pool)
            t0 = time.time()
            print("Started Burn-In")
            state = sampler.run_mcmc(p0, self.nburn, progress=True)
            print(f"Burn-In Took: {round((time.time() - t0) / 60, 2)}mins")
            sampler.reset()
            state = sampler.run_mcmc(state, self.niter, progress=True)

        DIR = 'data/pcaTrained'
        self.emulator_calls = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        self.gp.delete_folder_files("data/pcaTrained")
        self.state = state
        self.sampler = sampler
        self.niter_total = self.niter
        return None

    def retrain(self):

        self.state = self.sampler.run_mcmc(self.state, self.niter, progress=True)

        DIR = 'data/pcaTrained'
        self.emulator_calls = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        self.gp.delete_folder_files("data/pcaTrained")
        self.niter_total += self.niter

    def plot_corner(self):
        samples = np.copy(self.samples)
        samples = self.undo_normalization_samples(samples)
        # iobs_list = samples.reshape(-1, self.ndim)[3]
        # samples.reshape(-1, self.ndim)[3] = 90 - np.degrees(np.arccos(iobs_list / 10))
        corner.corner(samples.reshape(-1, self.ndim),  # collect samples into N x 3 array
                      bins=10,  # bins for histogram
                      show_titles=True, quantiles=[0.16, 0.84],  # show median and uncertainties
                      labels=self.labs,
                      truths=self.truth_arr,  # plot truth
                      color='darkviolet', truth_color='black',  # add some colors
                      **{'plot_datapoints': False, 'fill_contours': True})  # change some default options

    # def iobs_to_degrees(self, samples, index=3):
    #     iobs = samples.reshape(-1, self.ndim)[index]
    #     samples.reshape(-1, self.ndim)[index] = 90 - np.degrees(np.arccos(iobs / 10))
    #     return samples

    def plot_chains(self):
        samples = self.samples
        plt.figure(dpi=300, figsize=(10, 8))
        plt.subplot(4, 1, 1)
        [plt.plot(samples[:, i, 0], alpha=0.5) for i in range(self.nwalkers)]
        plt.xlim([0, self.niter])
        plt.xlabel('Iteration')
        plt.ylabel(self.labs[0])

        plt.subplot(4, 1, 2)
        [plt.plot(samples[:, i, 1], alpha=0.5) for i in range(self.nwalkers)]
        plt.xlabel('Iteration')
        plt.ylabel(self.labs[1])
        plt.xlim([0, self.niter])

        plt.subplot(4, 1, 3)
        [plt.plot(samples[:, i, 2], alpha=0.5) for i in range(self.nwalkers)]
        plt.xlim([0, self.niter])
        plt.xlabel('Iteration')
        plt.ylabel(self.labs[2])
        plt.tight_layout()

        plt.subplot(4, 1, 4)
        [plt.plot(samples[:, i, 3], alpha=0.5) for i in range(self.nwalkers)]
        plt.xlim([0, self.niter])
        plt.xlabel('Iteration')
        plt.ylabel(self.labs[3])
        plt.tight_layout()

    def plot_auto_correlation(self):
        def next_pow_two(n):
            i = 1
            while i < n:
                i = i << 1
            return i

        def auto_window(taus, c):
            m = np.arange(len(taus)) < c * taus
            if np.any(m):
                return np.argmin(m)
            return len(taus) - 1

        def autocorr_func_1d(x, norm=True):
            x = np.atleast_1d(x)
            if len(x.shape) != 1:
                raise ValueError("invalid dimensions for 1D autocorrelation function")
            n = next_pow_two(len(x))

            # Compute the FFT and then (from that) the auto-correlation function
            f = np.fft.fft(x - np.mean(x), n=2 * n)
            acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
            acf /= 4 * n

            # Optionally normalize
            if norm:
                acf /= acf[0]

            return acf

        # Following the suggestion from Goodman & Weare (2010)
        def autocorr_gw2010(y, c=5.0):
            f = autocorr_func_1d(np.mean(y, axis=0))
            taus = 2.0 * np.cumsum(f) - 1.0
            window = auto_window(taus, c)
            return taus[window]

        plt.figure(dpi=300)
        for dimension in range(len(self.initial)):
            chain = self.sampler.get_chain()[:, :, dimension].T
            N = np.exp(np.linspace(np.log(10), np.log(chain.shape[1]), 20)).astype(int)

            # Compute the estimators for a few different chain lengths
            gw2010 = np.empty(len(N))
            new = np.empty(len(N))
            for i, n in enumerate(N):
                gw2010[i] = autocorr_gw2010(chain[:, :n])
            #         new[i] = autocorr_new(chain[:, :n])

            # Plot the comparisons
            plt.loglog(N, gw2010, "o-", label=f"{self.labs[dimension]}")
        #     plt.loglog(N, new, "o-", label=f"New: {labs[dimension]}")

        ylim = plt.gca().get_ylim()
        plt.ylim(ylim)
        plt.xlabel("Number of Samples, $N$")
        plt.ylabel(r"$\tau$ Estimates")
        plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        plt.legend(fontsize=12)
        plt.title(f"Autocorrelation Efficiency")
        utkarshGrid()
        print(f"Walkers: {self.nwalkers}\nIterations: {self.niter_total}"
              f"\nEmulator Calls: {self.emulator_calls}\nTotal Runtime: {round((time.time() - self.t_init) / 60, 2)}mins")
        pass
