from Emulator.Classes.LightCurve import utkarshGrid, LightCurve
import numpy as np
import matplotlib.pyplot as plt
import GPy
from tqdm import tqdm
import pandas as pd


class GP(LightCurve):
    """The Gaussian Process for KNe Light Curves using GPy.
    """

    def __init__(self, referenceName):
        """ Instantiates class of both Gaussian Process and KNe Light Curve
        >>> gp = GP("reference.csv")
        >>> isinstance(gp, GP)
        True
        >>> isinstance(gp, LightCurve)
        True
        >>> isinstance(gp, float)
        False
        """
        LightCurve.__init__(self, referenceName)
        self.empirical = False
        self.isLog = False
        self.normalizeX = False
        self.extraction_time = 1
        self.failed = 0

    def range_select_wavelength(self, phi_range, mejdyn_range, mejwind_range, wv):
        """ Trauncate selection of light curve by wavelength.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.viewingangle.iloc[0][3]
        2.6453e-06
        >>> data.viewingangle.iloc[3][0]
        0.0019884
        >>> data.viewingangle.iloc[3][3]
        0.0029003
        >>> data.viewingangle.shape
        (100, 12)
        """
        self.select_viewingangle(phi_range, mejdyn_range, mejwind_range, wv)
        return None

    def single_time_step(self, time_of_interest, delta=0):
        """ Select single time step and save dataframe of viewing angles.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.time_sliced.shape
        (1, 11)
        >>> data.time_sliced.iloc[0][1]
        0.004539
        >>> data.single_time_step(2, delta = 2)
        >>> data.time_sliced.shape
        (5, 11)
        >>> data.time_sliced.iloc[1][1]
        0.0029853
        >>> data.time_sliced.iloc[0][1]
        0.0036462
        >>> data.time_sliced.iloc[1][0]
        0.0036834
        """
        time_of_int = time_of_interest
        time_ind = np.argmin(np.abs(self.time_arr - time_of_int))
        delt = delta
        day = self.viewingangle.iloc[time_ind - delt: time_ind + delt + 1]
        del day["time"]  # dont need time after choosing our time frame
        self.time_sliced = day
        return None

    def normedDF(self):
        """ Save the median normal of the dataframe.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.time_sliced_normed.shape
        (1, 11)
        >>> data.time_sliced_normed.iloc[0][1]
        -0.015038083458108309
        >>> data.time_sliced_normed.iloc[0][3]
        -0.07545081700410139
        >>> data.time_sliced_normed.iloc[0][6]
        0.12820345897619512
        """
        med = np.median(self.time_sliced)
        self.time_sliced_normed = self.time_sliced.divide(med) - 1
        return None

    def _normedArr(self, arr):
        """ Returns the median normal of the array.
        >>> data = GP("reference.csv")
        >>> newArr = np.array([0,1,2,3,4,5,6,7,8,9])
        >>> newArr2 = data._normedArr(newArr)
        >>> newArr3 = np.array([-1. , -0.77777778, -0.55555556, -0.33333333, -0.11111111,\
        0.11111111,  0.33333333,  0.55555556,  0.77777778,  1. ])
        >>> np.allclose(newArr2, newArr3)
        True
        >>> data.median
        4.5
        """
        med = np.median(arr)
        self.median = med
        return arr / med - 1

    def _undoNormedArr(self, arr):
        """ Undos the median normal of the array.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> np.isclose(data.medianCov, 2.123642889e-05)
        True
        >>> np.isclose(data.median, 0.0046083)
        True
        >>> data.median = data._undoNormedArr(data.time_sliced)
        >>> np.isclose(data.medianCov, 2.123642889e-05)
        True
        >>> np.isclose(float(data.median[0]), 0.004620670059690001)
        True
        >>> np.isclose(float(data.median[3]), 0.0046279341229800005)
        True
        """
        return (arr + 1) * self.median

    def _undoCovNorm(self, arr):
        """Undoes the normalization of the covariance matrix.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> np.isclose(data.medianCov, 2.123642889e-05)
        True
        >>> np.isclose(data.median, 0.0046083)
        True
        >>> preTest = np.random.rand(10,10)
        >>> testMat = data._undoCovNorm(preTest)
        >>> np.isclose(testMat[1,1], (preTest[1,1]+1)*data.medianCov)
        True
        >>> np.isclose(testMat[8,9], (preTest[8,9]+1)*data.medianCov)
        True
        >>> np.isclose(data.median, 0.0046083)
        True
        """
        return (arr + 1) * self.medianCov

    def setXY_viewingangle(self):
        """ Choose the X and Y training parameters by viewing angle.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.predX.shape
        (40, 1)
        >>> data.predX[3][0]
        0.8461538461538461
        >>> data.predX[-1][0]
        11.0
        """
        N = 40
        self.X = np.arange(0, self.Nobs, 1)
        self.Y = np.array(self.time_sliced_normed.iloc[0])
        self.X = self.X.reshape(len(self.X), 1)
        self.Y = self.Y.reshape(len(self.Y), 1)
        self.predX = np.linspace(0, self.Nobs, N).reshape(N, 1)
        return None

    def set_kernel(self, kernel):
        """Sets the kernel of the function
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> type(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        <class 'GPy.kern.src.rbf.RBF'>
        >>> type(data.kernel)
        <class 'GPy.kern.src.rbf.RBF'>
        """
        self.kernel = kernel
        return None

    def set_model(self, model):
        """Set the model for the gaussian process before training.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> type(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        <class 'GPy.models.gp_regression.GPRegression'>
        >>> type(data.model)
        <class 'GPy.models.gp_regression.GPRegression'>
        """
        self.model = model
        return None

    def set_predX(self, predX, include_like=False):
        """Sets the trianing data set.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> data.predX[0]
        array([0.])
        >>> data.predX[10]
        array([1.11111111])
        >>> data.predX[90]
        array([10.])
        >>> data.medianCov
        2.123642889e-05
        >>> data.median
        0.0046083
        """
        self.predX = predX
        #         currMean, currCov = self.model.predict(self.predX,  full_cov=True, include_likelihood = include_like)
        #         self._currMean, self._currCov = currMean, currCov
        self.median = np.median(self.time_sliced)
        self.medianCov = self.median ** 2
        return None

    def model_train(self, verbose=False, optimize_method="lbfgs"):
        """Model training
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.model.rbf.lengthscale[0]
        2.0
        >>> data.model_train()
        >>> data.model.rbf.lengthscale[0]
        5.104570976151195
        """
        if verbose:
            print(self.model)
        else:
            pass

        self.model.optimize(optimizer=optimize_method)

        #         self.model.optimize_restarts(verbose=False)

        if verbose:
            print(self.model.rbf.lengthscale)
        return None

    def plot_prior(self, manual=False, sig=1, randomDraws=True, title=None):
        """Plot the prior distribution with random draws. (Automatic plots the untrained posterior)
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> data.plot_prior(manual = True, sig = 2)
        >>> plt.close()
        >>> data.kernel.lengthscale[0]
        2.0
        >>> data.model.Gaussian_noise.variance[0]
        1.0
        >>> data.cov[0,0]
        2.0
        >>> data.cov[19,21]
        1.9876923466528822
        >>> data.cov[21,19]
        1.9876923466528822
        >>> data.cov[1,89]
        1.2910493813265383e-05
        """

        plotX = self.predX.reshape(1, len(self.predX))[0]

        if manual:
            predY_mean, prdY_cov = self.model.predict(self.predX, full_cov=True, include_likelihood=False)
            cov = self.kernel.K(self.predX)

        else:
            predY_mean, cov = self.model.predict(self.predX, full_cov=True, include_likelihood=False)

        var = np.diag(cov)
        mean_arr = predY_mean.reshape(1, len(predY_mean))[0]
        plotY = mean_arr

        F = np.random.multivariate_normal(mean_arr, cov, size=9)

        plt.figure(dpi=300)
        numVar = sig * np.sqrt(var)

        plt.fill_between(plotX, plotY + numVar, plotY - numVar, alpha=0.1, label=f"Confidence: {sig}"r"$\sigma$")
        plt.plot(plotX, plotY, label="Mean", color="Blue")

        if randomDraws:
            plt.plot(self.predX, F[0], color="red", linewidth=0.3, label="Random Draws")
            plt.plot(self.predX, F[1], color="red", linewidth=0.3)
            plt.plot(self.predX, F[2], color="red", linewidth=0.3)
            plt.plot(self.predX, F[3], color="red", linewidth=0.3)
            plt.plot(self.predX, F[4], color="red", linewidth=0.3)
            plt.plot(self.predX, F[5], color="red", linewidth=0.3)
            plt.plot(self.predX, F[6], color="red", linewidth=0.3)
            plt.plot(self.predX, F[7], color="red", linewidth=0.3)
            plt.plot(self.predX, F[8], color="red", linewidth=0.3)

        plt.xlabel(r"Viewing Angle ($\Phi$)")
        plt.ylabel("Deviation from Median")
        plt.scatter(self.X, self.Y, color="green", facecolors='none', label="Training Data")
        if title:
            plt.title(title)
        else:
            plt.title("GP Prior Distribution with random draws")
        utkarshGrid()
        plt.legend()
        self.cov = cov

    def plot_posterior(self, manual=False, sig=1, randomDraws=True, include_like=False):
        """ Plot trained data after optimizing gaussian process.
        >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> data.model_train(verbose = False)
        >>> data.plot_posterior(manual = True)
        >>> plt.close()
        >>> data.kernel.lengthscale[0]
        5.104570976151195
        >>> data.model.Gaussian_noise.variance[0]
        0.0343451657295366
        >>> data.cov[0,0]
        2.1414049353615357e-05
        >>> data.cov[19,21]
        2.1346097498941584e-05
        >>> data.cov[21,19]
        2.1346097498941584e-05
        >>> data.cov[1,89]
        2.122508391774732e-05
        >>> data.posterior_mean[0]
        0.003982024942205805
        >>> data.posterior_mean[50]
        0.004937711001749544
        >>> np.isclose((data.posterior_mean[-1])/data.median - 1, 0.1718569467329627)
        True
        """

        plotX = self.predX.reshape(1, len(self.predX))[0]

        if manual:
            trainX = self.X
            kernel = self.kernel
            noise = self.model.Gaussian_noise.variance
            bracket = kernel.K(trainX, trainX) + noise * np.identity(len(trainX))
            bracket_inv = np.linalg.inv(bracket)
            A = kernel.K(self.predX, trainX)
            B = bracket_inv
            C = kernel.K(trainX, self.predX)
            posterior_cov = kernel.K(self.predX, self.predX) - np.linalg.multi_dot([A, B, C])
            posterior_mean = np.linalg.multi_dot([kernel.K(self.predX, trainX), bracket_inv, self.Y])
            posterior_mean = posterior_mean.reshape(1, len(posterior_mean))[0]
            posterior_mean = np.array(posterior_mean, dtype=float)
            predY_mean = posterior_mean
            cov = posterior_cov

        else:
            self.kernel = self.kernel
            if include_like:
                predY_mean, cov = self.model.predict(self.predX, full_cov=True, include_likelihood=True)
            else:
                predY_mean, cov = self.model.predict(self.predX, full_cov=True, include_likelihood=False)

        self.unnormedY = self.Y
        cov = self._undoCovNorm(cov)
        predY_mean = self._undoNormedArr(predY_mean)
        self.unnormedY = self._undoNormedArr(self.Y)
        var = np.diag(cov)
        mean_arr = predY_mean.reshape(1, len(predY_mean))[0]
        plotY = mean_arr

        F = np.random.multivariate_normal(mean_arr, cov, size=3)

        plt.figure(dpi=300)
        numVar = sig * np.sqrt(var)

        plt.fill_between(plotX, plotY + numVar, plotY - numVar, alpha=0.1, label=f"Confidence: {sig}"r"$\sigma$")
        plt.plot(plotX, plotY, label="Mean", color="Blue")

        if randomDraws:
            plt.plot(self.predX, F[0], color="red", linewidth=0.3, label="Random Draws")
            plt.plot(self.predX, F[1], color="red", linewidth=0.3)
            plt.plot(self.predX, F[2], color="red", linewidth=0.3)

        plt.xlabel(r"Viewing Angle ($\Phi$)")
        plt.ylabel(r"Flux $Erg s^{-1} cm^{-2}A^{-1}$")
        plt.scatter(self.X, self.unnormedY, color="green", facecolors='none', label="Training Data")
        plt.title("GP Posterior Distribution with random draws")
        utkarshGrid()
        plt.legend()
        self.cov = cov
        self.posterior_mean = predY_mean

    def plot_covariance(self):
        """ Plot the current saved covariance matrix.
                >>> data = GP("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> data.model_train(verbose = False)
        >>> data.plot_posterior(manual = False)
        >>> plt.close()
        >>> data.plot_covariance()
        >>> plt.close()
        >>> np.isclose(data.cov[0,0], 2.1414049384491493e-05)
        True
        >>> np.isclose(data.cov[19,21], 2.1346097523916208e-05)
        True
        >>> np.isclose(data.cov[21,19], 2.1346097523916208e-05)
        True
        >>> np.isclose(data.cov[1,89], 2.122508391318888e-05)
        True
        >>> np.isclose(data.posterior_mean[0], 0.00398203)[0]
        True
        >>> np.isclose(data.posterior_mean[50], 0.00493771)[0]
        True
        >>> np.isclose(data.posterior_mean[-1], 0.00540027)[0]
        True
        """
        plt.figure(figsize=(4, 4), dpi=150)
        plt.imshow(self.cov, cmap="inferno", interpolation="none")
        plt.colorbar()
        plt.title(f"Covarance Matrix between {len(self.cov)} sampled points", fontsize=10)

    def LOOCV(self, manual=True, include_like=True):
        """Leave-One-Out Cross Validation of selected dataset. Cross validating by viewing angle.
        >>> data = GP("reference.csv")
        >>> phi_range = [45]
        >>> mejdyn_range = [0.01]
        >>> mejwind_range = [0.11]
        >>> wv = 900
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, wv)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> compareVA = data.viewingangle
        >>> kerLen = data.kernel.lengthscale[0]
        >>> compareM = data.model.Gaussian_noise.variance[0]
        >>> data.LOOCV()
        >>> np.isclose(data.looMean[0], 0.0025592363242248643)
        True
        >>> np.isclose(data.looMean[6], 0.00478646958190752)
        True
        >>> np.isclose(data.sigmaList[1], 0.00420134)[0]
        True
        >>> np.isclose(data.sigmaList[5], 0.00419738)[0]
        True
        >>> np.allclose(np.array(compareVA, dtype=float), np.array(data.viewingangle, dtype=float))
        True
        >>> np.isclose(data.looList[2], -0.072787432912073)
        True
        >>> np.isclose(data.looList[7], -0.04193157845918918)
        True
        >>> print(kerLen)
        2.0
        >>> np.isclose(data.kernel.lengthscale[0], 10.237731281822857)
        True
        >>> print(compareM)
        1.0
        >>> np.isclose(data.model.Gaussian_noise.variance[0], 0.019581516431695353)
        True

        """

        self.looMean = []
        self.sigmaList = []
        self.range_select_wavelength(self.phi_range, self.mejdyn_range, self.mejwind_range, self.wv_range)
        tempViewingAngle = self.viewingangle
        tempKernel = self.kernel
        tempModel = self.model
        tempX = self.X.copy()
        tempY = self.Y.copy()
        compareY = self.Y.copy()
        tempPredX = self.predX

        if manual:
            for i in range(self.Nobs):
                self.viewingangle = tempViewingAngle
                self.single_time_step(self.extraction_time)
                self.normedDF()
                self.setXY_viewingangle()
                self.set_kernel(tempKernel)
                self.set_model(tempModel)
                test_pointX = np.array([tempX[i]])
                test_pointY = np.array([tempY[i]])
                self.X = np.delete(tempX, i)
                self.X = self.X.reshape(len(self.X), 1)
                self.Y = np.delete(tempY, i)
                self.set_predX(tempPredX)
                self.model_train(verbose=False)
                self.X = tempX

                mean, var = self.model.predict(test_pointX, include_likelihood=include_like)
                var = self._undoCovNorm(var)
                sigma = np.sqrt(var)
                mean, sigma = mean[0], sigma[0]
                mean = self._undoNormedArr(mean)[0]
                self.looMean.append(mean)
                self.sigmaList.append(sigma)

            tempY.T[0] = self._undoNormedArr(tempY.T[0])
            tempY.T[0] = np.array(tempY.T[0], dtype=float)

            compareY = np.array(compareY, dtype=float)
            tempY = np.array(tempY, dtype=float)

            self.Y = compareY

            arr1 = np.array(tempY.T[0], dtype=float)
            arr2 = np.array(self.time_sliced.to_numpy()[0], dtype=float)
            assert (np.allclose(arr1, arr2))
            self.looList = (self.looMean - tempY.T[0]) / np.array(self.sigmaList).T[0]

        else:
            gp = self
            gp.normedDF()
            gp.setXY_viewingangle()
            gp.set_predX(self.set_predX(self.predX))
            var, mean = gp.model.predict(gp.X)
            gp.model_train(verbose=False)
            var, mean = gp.model.predict(gp.X)
            var = gp._undoCovNorm(var)

            # Calculate LOO
            loos = gp.model.inference_method.LOO(gp.kernel, gp.X, gp.Y, gp.model.likelihood, gp.model.posterior)
            loo_error = np.sum(loos)
            print(f"Leave one out density: {loo_error}")
            plt.figure(figsize=(6, 3), dpi=300)
            plt.scatter(gp.X, gp._undoNormedArr((gp.Y - loos) / np.sqrt(var)), facecolor="none", edgecolor="dodgerblue")
            plt.xlabel("Viewing Angle Left Out")
            plt.ylabel(r"Leave-One-Out Accuracy $\sigma$")
            plt.title("GPy LOO (Possibly not in Sigma Units)")
            utkarshGrid()

        self.viewingangle = tempViewingAngle
        self.set_kernel(tempKernel)
        self.set_model(tempModel)
        return None

    def plot_loocv(self, plot_type="single"):
        """ Plot of the errors in the LOO optimization.
        >>> data = GP("reference.csv")
        >>> phi_range = [45]
        >>> mejdyn_range = [0.01]
        >>> mejwind_range = [0.11]
        >>> wv = 900
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, wv)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> data.LOOCV()
        >>> data.plot_loocv()
        >>> plt.close()
        >>> np.isclose(data.plot_loocv_limitY, 0.2381935005242566)
        True
        """
        if plot_type == "multiple":
            self.plotLooList = np.mean(self.loo_list_multiple, axis=0)
        elif plot_type == "single":
            self.plotLooList = self.looList
        else:
            print("[ERROR] Plot type not selected")

        plt.figure(figsize=(6, 3), dpi=300)
        plt.scatter(self.X.T[0], self.plotLooList, facecolor="none", edgecolor="dodgerblue")
        plt.xlabel("Viewing Angle Left Out")
        plt.ylabel(r"Accuracy (Units $\sigma$)")
        plt.title("Leave-One-Out Cross Validation")
        utkarshGrid()
        limitY = max(abs(min(self.plotLooList)) * 1.1, abs(max(self.plotLooList)) * 1.1)
        self.plot_loocv_limitY = limitY
        plt.ylim(-limitY, limitY)
        return None

    def plot_loocv_simple(self, include_like=True):
        """ Simple Plot of the LOOCV on the posterior distribution.
        >>> data = GP("reference.csv")
        >>> phi_range = [45]
        >>> mejdyn_range = [0.01]
        >>> mejwind_range = [0.11]
        >>> wv = 900
        >>> data.range_select_wavelength(phi_range, mejdyn_range, mejwind_range, wv)
        >>> data.single_time_step(1)
        >>> data.normedDF()
        >>> data.setXY_viewingangle()
        >>> data.set_kernel(GPy.kern.RBF(input_dim=1, variance = 2, lengthscale=2))
        >>> data.set_model(GPy.models.GPRegression(data.X, data.Y, data.kernel))
        >>> data.set_predX(np.linspace(0,data.Nobs,100).reshape(100, 1))
        >>> data.LOOCV()
        >>> data.plot_loocv_simple()
        >>> plt.close()
        >>> data.model.Gaussian_noise.variance[0]
        0.019581516431695353
        >>> data.cov[0,0]
        1.7704562471056736e-05
        >>> data.cov[19,21]
        1.7286183692651927e-05
        >>> data.cov[21,19]
        1.7286183692651927e-05
        >>> data.cov[1,89]
        1.722470910238539e-05
        >>> np.isclose(data.posterior_mean[0], 0.00255924)[0]
        True
        >>> np.isclose(data.posterior_mean[50], 0.00459252)[0]
        True
        """
        self.plot_posterior(include_like=True)
        plt.scatter(self.X, self.looMean, label="Loo Prediction", color="black", marker='X', zorder=1)
        plt.legend()
        return None

    def multiple_LOOCV(self, N=3, verbose=True):
        """ Multiple Gaussian Processes on different datasets to apply Leave-One-Out CV.
        >>> data = GP("reference.csv")
        >>> data.multiple_LOOCV(N = 5, verbose = False)
        >>> data.loo_list_multiple.shape
        (5, 11)
        >>> data.phi_range_list
        [45, 15, 75, 75, 60]
        >>> data.mejdyn_range_list
        [0.001, 0.01, 0.001, 0.005, 0.02]
        >>> data.wv_range_list
        [900, 900, 900, 900, 900]
        >>> np.isclose(data.loo_list_multiple[0,1], 4.767686375520658e-10)
        True
        >>> np.isclose(data.loo_list_multiple[1,0], 0.06800734333543058)
        True
        >>> np.isclose(data.loo_list_multiple[4,3], 0.03651750786814857)
        True
        >>> np.isclose(data.kernel.lengthscale[0], 5.104570976151195)
        True
        >>> np.isclose(data.model.Gaussian_noise.variance[0], 0.0343451657295366)
        True
        """
        self.phi_range_list = list(self.reference.phi)[0:N]
        self.mejdyn_range_list = list(self.reference.mejdyn)[0:N]
        self.mejwind_range_list = list(self.reference.mejwind)[0:N]
        self.wv_range_list = [900] * N

        self.loo_list_multiple = []
        for i in tqdm(range(N), disable=not verbose):
            tempSelected = self.selected

            self.range_select_wavelength([self.phi_range_list[i]],
                                         [self.mejdyn_range_list[i]],
                                         [self.mejwind_range_list[i]],
                                         self.wv_range_list[i])
            if self.Nobs != 11:
                if verbose:
                    print(
                        f"[STATUS] File selected at {self.selected.filename.iloc[0]} \ndoes not have the correct number of viewing angles. Skipping...")
                self.selected = tempSelected
                continue
            self.single_time_step(self.extraction_time)
            self.normedDF()
            self.setXY_viewingangle()
            self.set_kernel(GPy.kern.RBF(input_dim=1, variance=2, lengthscale=2))
            self.set_model(GPy.models.GPRegression(self.X, self.Y, self.kernel))
            self.LOOCV()
            self.loo_list_multiple.append(self.looList)
            self.selected = tempSelected

        self.loo_list_multiple = np.array(self.loo_list_multiple)
        return None

    def gaussian(self, x, x0, sigma):
        return np.exp(-np.power((x - x0) / sigma, 2.) / 2.)

    def plot_loocv_histogram(self, edge=2.5, mu=0, sigma=1, binning=30):
        fig, ax = plt.subplots(dpi=300)
        utkarshGrid()

        hist_arr = self.loo_list_multiple.flatten()
        hist_arr = hist_arr[np.isfinite(hist_arr)]
        hist_arr = hist_arr[hist_arr < 3]
        hist_arr = hist_arr[hist_arr > -3]
        print(f"Inside 3x: {len(hist_arr)}, Total: {len(self.loo_list_multiple.flatten())}")

        df = pd.DataFrame(hist_arr, columns=["hist"])

        if not self.empirical:
            x_gauss = np.linspace(-edge, edge, 100, endpoint=True)
            y_gauss = self.gaussian(x_gauss, mu, sigma)
            plt.plot(x_gauss, y_gauss, label="Unit Gaussian", color="purple", zorder=3)

        df.plot.hist(density=True, bins=binning, ax=ax, label="Count",
                     facecolor='#2ab0ff', edgecolor='#169acf', zorder=1)
        df.plot.kde(ax=ax, label="LOO Distribution", alpha=1, zorder=2)
        plt.ylabel("Count Intensity")

        if self.empirical:
            ax.legend(["LOO Distribution ", "Count"])
            ax.set_title(r"Ratio = $\frac{Truth - Predictive}{Truth}$")

            if self.isLog:
                plt.xlabel(r"Deviation Error (Units Log Flux)")
            else:
                plt.xlabel(r"Deviation Error (Units Flux)")
        else:
            plt.xlabel(r"Deviation Error (Units $\sigma$)")
            ax.legend(["Unit Gaussian", "LOO Distribution ", "Count"])
            ax.set_title(r"Ratio = $\frac{{Truth - Predictive}}{\sigma}$")

        ax.set_ylim(bottom=-0.1)

    def plot_hist_lengthscale(self, arr_hist, typ0=None, tol=5):
        """ Plots the histogram distribution of the lengthscales.
        """
        temp_hist = arr_hist
        arr_hist = arr_hist[arr_hist < tol]

        if not typ0:
            typ0 = "UNSPECIFIED"

        print(f"Dimension {typ0}: {round(100 * len(arr_hist) / len(temp_hist))}% within lengthscale {tol}.")

        fig, ax = plt.subplots(dpi=100)
        hist_arr = arr_hist.flatten()
        utkarshGrid()
        df = pd.DataFrame(hist_arr, columns=["hist"])
        df.plot.hist(density=True, bins=20, ax=ax, label="Count",
                     facecolor='#23de6b', edgecolor='#18b855', zorder=1)

        try:
            df.plot.kde(ax=ax, label="Lengthscale Distribution", alpha=1, zorder=2)

        except:
            if np.max(arr_hist) == np.min(arr_hist):
                print(f"[WARNING] Are all values of this array are the same!")
            else:
                print("[ERROR] Something went wrong")
        plt.ylabel("Count Intensity")
        ax.legend(["Lengthscale Distribution", "Count"])
        plt.xlabel(r"Lengthscale")
        ax.set_ylim(bottom=-1e-9)
        return None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
