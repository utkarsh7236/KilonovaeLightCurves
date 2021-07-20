from Emulator.Classes.LightCurve import utkarshGrid, LightCurve
from Emulator.Classes.GP import GP
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import GPy
import time
from joblib import Parallel, delayed
from itertools import product


class GP2D(GP):
    """The Gaussian Process for KNe Light Curves using GPy. This class handles the two dimensional case.
    """

    def __init__(self, referenceName):
        """ Instantiates class of both Gaussian Process and KNe Light Curve
        >>> gp = GP2D("reference.csv")
        >>> isinstance(gp, GP)
        True
        >>> isinstance(gp, LightCurve)
        True
        >>> isinstance(gp, float)
        False
        >>> isinstance(gp, GP2D)
        True
        """
        GP.__init__(self, referenceName)
        return None

    def set_selection_range(self, typ, phi_range=[], mejwind_range=[], mejdyn_range=[], wv_range=[900], verbose=False):
        """
        Selection method that converts the range desired into a two dimensional training vector.
        First dimension is the viewing angle, second dimension is selected type.
        The data vector returned is not normalized.

        >>> data = GP2D("reference.csv")
        >>> phi_range = [60]
        >>> mejwind_range = [0.05]
        >>> mejdyn_range = [0.01]
        >>> wv_range = [1500]
        >>> data.set_selection_range(typ = "mejdn", phi_range = phi_range, mejwind_range = mejwind_range, mejdyn_range = mejdyn_range,wv_range = wv_range, verbose = False)
        [ERROR] Incorrect selection of 2D parameters. Try Again...
        >>> data = GP2D("reference.csv")
        >>> phi_range = [60]
        >>> mejwind_range = [0.05]
        >>> mejdyn_range = [0.01]
        >>> wv_range = [1500]
        >>> data.set_selection_range(typ = "mejwind", phi_range = phi_range, mejwind_range = mejwind_range, mejdyn_range = mejdyn_range,wv_range = wv_range, verbose = False)
        """
        self.typ = typ
        self.wv_range = wv_range
        self.phi_range = phi_range
        self.mejwind_range = mejwind_range
        self.mejdyn_range = mejdyn_range
        self.wv = self.wv_range[0]

        if typ == "mejdyn":
            self.mejdyn_range = []

        elif typ == "mejwind":
            self.mejwind_range = []

        elif typ == "phi":
            self.phi_range = []

        else:
            print("[ERROR] Incorrect selection of 2D parameters. Try Again...")
            return None

        self.select_curve(phiRange=self.phi_range,
                          mejwindRange=self.mejwind_range,
                          mejdynRange=self.mejdyn_range)

        if typ == "mejdyn":
            curr_range_list = self.selected.mejdyn.unique()

        elif typ == "mejwind":
            curr_range_list = self.selected.mejwind.unique()

        elif typ == "phi":
            curr_range_list = self.selected.phi.unique()

        else:
            curr_range_list = None
            print("[ERROR] Incorrect selection of 2D parameters. Try Again...")

        self.curr_range_list = curr_range_list
        self.curr_range_list = np.sort(self.curr_range_list)

        df = None
        temp_curr_range_list = curr_range_list
        for i in tqdm(range(len(curr_range_list)), disable=not verbose):
            tempSelected = self.selected

            if typ == "mejdyn":
                self.mejdyn_range = [curr_range_list[i]]

            elif typ == "mejwind":
                self.mejwind_range = [curr_range_list[i]]

            elif typ == "phi":
                self.phi_range = [curr_range_list[i]]

            else:
                print("[ERROR] Incorrect selection of 2D parameters. Try Again...")
                return None

            self.range_select_wavelength(self.phi_range, self.mejdyn_range, self.mejwind_range, self.wv)

            #             if self.Nobs != 11:
            #                 if verbose:
            #                     print(f"[ERROR] File selected at {self.selected.filename.iloc[0]} \ndoes not have the correct number of viewing angles. Skipping...")
            #                 self.selected = tempSelected
            #                 temp_curr_range_list = np.delete(temp_curr_range_list, i)
            #                 continue

            self.single_time_step(self.extraction_time)  # Want distribution at 1 day
            temp_df = self.time_sliced
            df = pd.concat([df, temp_df])
            self.selected = tempSelected

        if df.shape[1] == 1:
            repeats = 11
            df = pd.concat([df] * repeats, axis=1)
            self.Nobs = 11

        curr_range_list = temp_curr_range_list
        df.index = curr_range_list
        df.index.name = self.typ
        df.reset_index(drop=True)
        df.fillna(axis=1, method='ffill', inplace=True)
        self.Nobs = 11
        epsilon = 1e-15
        self.iobs_range = np.linspace(0, self.Nobs - 1, self.Nobs, endpoint=True)
        df = df.add(epsilon)
        self.training2D = df.sort_index()
        self.curr_range_list = curr_range_list
        self.curr_range_list = np.sort(self.curr_range_list)
        return None

    def normalize_training2D(self):
        """ Normalize the two dimensional training vector using median normalization.
        >>> data = GP2D("reference.csv")
        >>> phi_range = [60]
        >>> mejwind_range = [0.05]
        >>> mejdyn_range = [0.01]
        >>> wv_range = [1500]
        >>> data.set_selection_range(typ = "mejdyn", phi_range = phi_range, mejwind_range = mejwind_range, mejdyn_range = mejdyn_range,wv_range = wv_range, verbose = False)
        >>> data.normalize_training2D()
        >>> checkDF = data.training2D
        >>> mat1 = checkDF.to_numpy(dtype = float)
        >>> mat2 = data._undoNormedDF(data.training2D_normalized).to_numpy(dtype = float)
        >>> np.allclose(mat1, mat2)
        True
        """
        self.training2D_normalized = self._normedDF(self.training2D)
        return None

    def unnormalize_training2D(self, verbose=True):
        """ Normalize the two dimensional training vector using median normalization.
        >>> gp = GP2D("reference.csv")
        >>> phi_range = [30]
        >>> mejwind_range = [0.05]
        >>> mejdyn_range = [0.01]
        >>> wv_range = [900]
        >>> gp.set_selection_range(typ = "mejwind", phi_range = phi_range,mejwind_range = mejwind_range, mejdyn_range = mejdyn_range, wv_range = wv_range, verbose = False)
        >>> gp.setXY()
        >>> gp.set_kernel(GPy.kern.RBF(input_dim=2, variance = 1, lengthscale=10))
        >>> gp.set_model(GPy.models.GPRegression(gp.X,gp.Y,gp.kernel))
        >>> gp.model_train(verbose = False)
        >>> gp.model_predict(N = 5)
        >>> gp.unnormalize_training2D()
        [ERROR] Data has not been normalized, so it cannot be unnormalized
        """

        if verbose:
            try:
                self.median
            except:
                print("[ERROR] Data has not been normalized, so it cannot be unnormalized")
                return None

        self.posterior_mean = self._undoNormedArr(self.posterior_mean)
        self.posterior_cov = self._undoCovNorm(self.posterior_cov)
        return None

    def _normedDF(self, df):
        med = np.median(df)
        self.median = med
        self.medianCov = self.median ** 2
        return df.divide(med) - 1

    def _undoNormedDF(self, df):
        return (df + 1).multiply(self.median)

    def _check_normalization_fixY(self):
        try:
            self.training2D_normalized
        except:
            self.training2D_normalized = self.training2D

        return None

    def setXY(self):
        """ Set X and Y training vectors in the format of GPy

        >>> data = GP2D("reference.csv")
        >>> phi_range = [60]
        >>> mejwind_range = [0.05]
        >>> mejdyn_range = [0.01]
        >>> wv_range = [1500]
        >>> data.set_selection_range(typ = "mejdyn", phi_range = phi_range, mejwind_range = mejwind_range, mejdyn_range = mejdyn_range,wv_range = wv_range, verbose = False)
        >>> data.normalize_training2D()
        >>> X1 = data.iobs_range
        >>> X2 = np.array(data.curr_range_list)
        >>> Y = data.training2D_normalized.to_numpy(dtype = float).flatten()
        >>> Y = Y.reshape(len(Y), 1)
        >>> _X1, _X2 = np.meshgrid(X1, X2)
        >>> _X1 = _X1.flatten()
        >>> _X1 = _X1.reshape(len(_X1), 1)
        >>> _X2 = _X2.flatten()
        >>> _X2 = _X2.reshape(len(_X2), 1)
        >>> X = np.hstack([_X1, _X2])
        >>> data.setXY()
        >>> np.allclose(X, data.X)
        True
        >>> np.allclose(Y, data.Y)
        True
        >>> np.allclose(X1, data.X1)
        True
        >>> np.allclose(X2, data.X2)
        True
        """
        X1 = self.iobs_range
        X2 = np.array(self.curr_range_list)

        if self.normalizeX:
            normalizedX1 = (X1 - min(X1)) / (max(X1) - min(X1))
            normalizedX2 = (X2 - min(X2)) / (max(X2) - min(X2))

            X1, X2 = normalizedX1, normalizedX2

        self._check_normalization_fixY()

        Y = self.training2D_normalized.to_numpy(dtype=float)

        Y = Y.flatten()
        Y = Y.reshape(len(Y), 1)
        _X1, _X2 = np.meshgrid(X1, X2)
        _X1 = _X1.flatten()
        _X1 = _X1.reshape(len(_X1), 1)
        _X2 = _X2.flatten()
        _X2 = _X2.reshape(len(_X2), 1)
        X = np.hstack([_X1, _X2])

        self.Y = Y
        self.X1 = X1
        self.X2 = X2
        self.X = X
        return None

    def model_predict(self, N=50, include_like=True, make_cov=True, same_dimension=False):
        curr_range_list = self.curr_range_list

        PredX1 = np.linspace(min(self.iobs_range), max(self.iobs_range), N, endpoint=True)
        PredX2 = np.linspace(min(curr_range_list), max(curr_range_list), N, endpoint=True)

        if self.normalizeX:
            normalizedX1 = (PredX1 - min(PredX1)) / (max(PredX1) - min(PredX1))
            normalizedX2 = (PredX2 - min(PredX2)) / (max(PredX2) - min(PredX2))

            PredX1, PredX2 = normalizedX1, normalizedX2

        _X1Pred, _X2Pred = np.meshgrid(PredX1, PredX2)
        _X1Pred = _X1Pred.flatten()
        _X1Pred = _X1Pred.reshape(len(_X1Pred), 1)
        _X2Pred = _X2Pred.flatten()
        _X2Pred = _X2Pred.reshape(len(_X2Pred), 1)
        predX = np.hstack([_X1Pred, _X2Pred])

        mean, cov = self.model.predict(predX, full_cov=make_cov, include_likelihood=include_like)
        self.posterior_mean = mean
        self.posterior_cov = cov
        self.predX1 = PredX1
        self.predX2 = PredX2
        self.predX = predX
        self.N = N
        return None

    def plot_covariance2D(self):
        plt.figure(figsize=(4, 4), dpi=150)
        plt.imshow(self.posterior_cov, cmap="inferno", interpolation="none")
        plt.colorbar()
        plt.title(f"Covarance Matrix between {len(self.posterior_cov)} sampled points", fontsize=10)
        return None

    def plot_posterior2D(self, verbose=False, lev=20):
        plt.figure(dpi=200)
        plt.tight_layout()
        Z = self.posterior_mean.reshape(self.N, self.N)
        contours = plt.contourf(self.predX1, self.predX2, Z, cmap="plasma", levels=lev)
        # plt.clabel(contours, inline=True, fontsize=9, colors = "white")
        plt.colorbar()
        plt.xlabel("Viewing Angle")

        if self.typ == "mejdyn":
            plt.ylabel("Dynamical Ejecta Mass")

        elif self.typ == "mejwind":
            plt.ylabel("Wind Ejecta Mass")

        elif self.typ == "phi":
            plt.ylabel(r"Half Opening Angle $\Phi$")

        else:
            print("[ERROR] Incorrect selection of 2D parameters. Try Again...")

        if verbose:
            if self.typ == "mejdyn":
                print(
                    f"[STATUS] Plotting for: \n[STATUS] mejdyn: {self.curr_range_list} \n[STATUS] mejwind: {self.mejwind_range} \n[STATUS] phi: {self.phi_range} \n[STATUS] viewing_angle: {self.iobs_range} \n[STATUS] wavelength: {self.wv_range}")

            elif self.typ == "mejwind":
                print(
                    f"[STATUS] Plotting for: \n[STATUS] mejdyn: {self.mejdyn_range} \n[STATUS] mejwind: {self.curr_range_list} \n[STATUS] phi: {self.phi_range} \n[STATUS] viewing_angle: {self.iobs_range} \n[STATUS] wavelength: {self.wv_range}")

            elif self.typ == "phi":
                print(
                    f"[STATUS] Plotting for: \n[STATUS] mejdyn: {self.mejdyn_range} \n[STATUS] mejwind: {self.mejwind_range} \n[STATUS] phi: {self.curr_range_list} \n[STATUS] viewing_angle: {self.iobs_range} \n[STATUS] wavelength: {self.wv_range}")
        return None

    def log_trainingND(self):
        """
        >>> data = GP2D("reference.csv")
        >>> phi_range = [45]
        >>> mejwind_range = []
        >>> mejdyn_range = [0.01]
        >>> wv_range = [900]
        >>> data.set_selection_range(typ = "mejwind", phi_range = phi_range, mejwind_range = mejwind_range, mejdyn_range = mejdyn_range, wv_range = wv_range, verbose = False)
        >>> arr1 = np.log10(data.training2D.to_numpy(dtype = float))
        >>> data.log_trainingND()
        >>> arr2 = data.training2D.to_numpy(dtype = float)
        >>> np.allclose(arr1, arr2)
        True
        """
        self.training2D = self.training2D.applymap(np.log10)
        self.isLog = True
        return None

    def set_normalizeX(self):
        self.normalizeX = True
        return None

    def LOOCV_2D(self, include_like=True, make_cov=False, verbose=True):
        """ Leave one out cross-validation for two-dimensional case.
        >>> import warnings
        >>> warnings.filterwarnings("ignore")
        >>> data = GP2D("reference.csv")
        >>> phi_range = [45]
        >>> mejwind_range = [0.03]
        >>> mejdyn_range = [0.01]
        >>> wv_range = [900]
        >>> data.set_selection_range(typ = "mejwind", phi_range = phi_range, mejwind_range = mejwind_range, mejdyn_range = mejdyn_range, wv_range = wv_range, verbose = False)
        >>> data.kernel = GPy.kern.RBF(input_dim=2, variance = 1, lengthscale=10, ARD = True)
        >>> data.LOOCV_2D(verbose = False)
        (True, 0)
        >>> np.isclose(data.Y.T[0][20],  -0.03469556751158931)
        True
        >>> np.isclose(data.Y.T[0][60],  -0.06234175674452824)
        True
        >>> np.isclose(data.looList[30], -0.01661552)[0][0]
        True
        >>> np.isclose(data.looList[50], -0.04954337)[0][0]
        True
        >>> old_med = data.median
        >>> np.isclose(old_med, -2.447331783887685)
        True
        >>> data.tempY = (data.tempY + 1)*old_med
        >>> arr1 = np.array(data.tempY.T[0], dtype=float)
        >>> arr2 = np.array(data.training2D.to_numpy().flatten(), dtype=float)
        >>> np.allclose(arr1, arr2)
        True
        >>> np.isclose(data.model.rbf.lengthscale[0], 0.3643163918154837)
        True
        >>> np.isclose(data.model.rbf.lengthscale[1], 0.1780219066605998)
        True
        """
        failed = 0
        self.log_trainingND()
        tempTraining2D = self.training2D

        # Begin of example step
        self.normalize_training2D()
        self.set_normalizeX()
        self.setXY()
        self.set_model(GPy.models.GPRegression(self.X, self.Y, self.kernel))

        # Begin of LOO
        self.looMean = []
        self.sigmaList = []
        tempKernel = self.kernel.copy()
        tempModel = self.model.copy()
        originalX = self.X.copy()
        originalY = self.Y.copy()
        tempX = self.X.copy()
        tempY = self.Y.copy()
        self.looList = []
        self.looList_empirical = []
        self.lengthscaleList = []
        for i in tqdm(range(len(tempX)), disable=not verbose):
            test_pointX = np.array([tempX[i]])
            test_pointY = np.array([tempY[i]])
            self.X = np.delete(tempX, i, 0)
            self.Y = np.delete(tempY, i, 0)
            self.set_kernel(tempKernel.copy())
            self.set_model(GPy.models.GPRegression(self.X, self.Y, self.kernel))
            # self.model['.*lengthscale'].constrain_bounded(0,5)
            try:
                self.model_train(verbose=False, optimize_method="lbfgs")
            except:
                failed += 1
                continue
            #             print(self.model.rbf.lengthscale[0], self.model.rbf.lengthscale[1])
            self.lengthscaleList.append([self.model.rbf.lengthscale[0], self.model.rbf.lengthscale[1]])
            mean, var = self.model.predict(test_pointX, full_cov=make_cov, include_likelihood=include_like)
            mean = self._undoNormedArr(mean)[0]
            var = self._undoCovNorm(var)
            sigma = np.sqrt(var)
            mean, sigma = mean[0], sigma[0][0]

            test_pointY = (test_pointY + 1) * self.median  # Unnormalize
            difference = mean - test_pointY

            self.looList.append(difference / sigma)
            self.looList_empirical.append(difference / test_pointY)

            #             self.looMean.append(mean)
            #             self.sigmaList.append(sigma)
            self.X, tempX = originalX.copy(), originalX.copy()
            self.Y, tempY = originalY.copy(), originalY.copy()

        self.looList = np.array(self.looList, dtype=float)
        self.looList_empirical = np.array(self.looList_empirical, dtype=float)
        #         self.set_kernel(tempKernel)
        #         self.set_model(tempModel)
        self.tempY = tempY
        self.originalY = originalY
        return self.isLog, failed

    def multiple_LOOCV_2D(self, typ, verbose=True, trauncate=None, include_like=True, empirical=False):
        """
        >>> data = GP2D("reference.csv")
        >>> ref = data.reference
        >>> typ = "phi"
        >>> data.phi_range = [45]
        >>> data.mejdyn_range = [0.01]
        >>> data.mejwind_range = [0.05]
        >>> data.wv_range = [900]
        >>> data.kernel = GPy.kern.RBF(input_dim=2, variance = 1, lengthscale=10, ARD = True)
        >>> data.multiple_LOOCV_2D(typ, verbose = 0, trauncate = 2)
        >>> data.empirical
        False
        >>> arr1 = data.loo_list_multiple.flatten()
        >>> arr1 = arr1[~np.isnan(arr1)]
        >>> np.isclose(arr1[0], 0.010368926670178167, atol = 1e-04)
        True
        >>> np.isclose(arr1[54], 0.10859439911838269)
        True
        >>> data.isLog
        True
        >>> data = GP2D("reference.csv")
        >>> ref = data.reference
        >>> typ = "phi"
        >>> data.phi_range = [45]
        >>> data.mejdyn_range = [0.01]
        >>> data.mejwind_range = [0.05]
        >>> data.wv_range = [900]
        >>> data.kernel = GPy.kern.RBF(input_dim=2, variance = 1, lengthscale=10, ARD = True)
        >>> data.multiple_LOOCV_2D(typ, verbose = 0, trauncate = 2, empirical = True)
        >>> arr2 = data.loo_list_multiple.flatten()
        >>> arr2 = arr2[~np.isnan(arr2)]
        >>> np.isclose(arr2[0], -0.01352243495614905, atol = 1e-04)
        True
        >>> np.isclose(arr2[54],  -0.10777332196137197)
        True
        >>> print(arr2.shape)
        (154,)
        >>> ref.equals(data.reference)
        True
        >>> ref.equals(data.selected)
        True
        >>> data.phi_range
        []
        >>> data.wv_range
        [900]
        >>> data.mejwind_range
        [0.05]
        >>> data.mejdyn_range
        [0.01]
        >>> data.empirical
        True
        >>> data.isLog
        True
        >>> data.lengthscaleList_multiple1.shape
        (154,)
        >>> data.lengthscaleList_multiple2.shape
        (154,)
        """
        self.loo_list_multiple = np.array([], dtype=float)
        self.lengthscaleList_multiple1 = np.array([], dtype=float)
        self.lengthscaleList_multiple2 = np.array([], dtype=float)

        if typ == "mejwind":
            a = self.reference.phi.unique()
            b = self.reference.mejdyn.unique()
            self.mejwind_range = []

        if typ == "phi":
            a = self.reference.mejwind.unique()
            b = self.reference.mejdyn.unique()
            self.phi_range = []

        if typ == "mejdyn":
            a = self.reference.mejwind.unique()
            b = self.reference.phi.unique()
            self.mejdyn_range = []

        curr_pair = list(product(a, b))
        self.counter = 0
        wv_range = self.wv_range

        if trauncate is None:
            loop_length = len(curr_pair)

        else:
            loop_length = trauncate

        #         for i in tqdm(range(loop_length), disable = disableTQDM):
        #             pass

        def parallel_helper(self, i):
            if typ == "mejwind":
                phi_range = [curr_pair[i][0]]
                mejwind_range = self.selected.mejwind.unique()
                mejdyn_range = [curr_pair[i][1]]

            if typ == "phi":
                phi_range = self.selected.phi.unique()
                mejwind_range = [curr_pair[i][0]]
                mejdyn_range = [curr_pair[i][1]]

            if typ == "mejdyn":
                phi_range = [curr_pair[i][1]]
                mejwind_range = [curr_pair[i][0]]
                mejdyn_range = self.selected.mejdyn.unique()

            self.set_selection_range(typ=typ, phi_range=phi_range,
                                     mejwind_range=mejwind_range,
                                     mejdyn_range=mejdyn_range,
                                     wv_range=wv_range, verbose=False)
            self.counter += 1

            if verbose > 10:
                looCV_verbose = True
            else:
                looCV_verbose = False

            self.isLog, self.failed = self.LOOCV_2D(verbose=looCV_verbose, include_like=include_like)

            if empirical:
                self.loo_list_multiple = np.append(self.loo_list_multiple, self.looList_empirical)

            else:
                self.loo_list_multiple = np.append(self.loo_list_multiple, self.looList)

            self.selected = self.reference

            arr1 = np.array([x[0] for x in self.lengthscaleList], dtype=float)
            arr2 = np.array([x[1] for x in self.lengthscaleList], dtype=float)
            arr1 = arr1.flatten()
            arr2 = arr2.flatten()
            self.lengthscaleList_multiple1 = np.append(self.lengthscaleList_multiple1, arr1)
            self.lengthscaleList_multiple2 = np.append(self.lengthscaleList_multiple2, arr2)

            self.loo_list_multiple = np.array(self.loo_list_multiple, dtype=float)
            return self.loo_list_multiple, self.counter, self.lengthscaleList_multiple1, self.lengthscaleList_multiple2, self.isLog, self.failed

        self.results = Parallel(n_jobs=8, verbose=verbose)(
            delayed(parallel_helper)(self, i) for i in range(loop_length))

        for x in self.results:
            self.loo_list_multiple = np.append(self.loo_list_multiple, x[0])
            self.lengthscaleList_multiple1 = np.append(self.lengthscaleList_multiple1, x[2])
            self.lengthscaleList_multiple2 = np.append(self.lengthscaleList_multiple2, x[3])

        self.counter = sum(np.array([x[1] for x in self.results], dtype=int))
        temp = np.array([x[4] for x in self.results], dtype=float)
        self.isLog = np.all(temp)
        self.failed = np.array([x[5] for x in self.results], dtype=int)

        self.empirical = empirical

        time.sleep(0.5)
        if verbose > 0:
            print(f"[STATUS] Used {self.counter}/{len(curr_pair)} items")

        return None