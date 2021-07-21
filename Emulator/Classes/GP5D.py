from Emulator.Classes.LightCurve import utkarshGrid, LightCurve
from Emulator.Classes.GP import GP
from Emulator.Classes.GP2D import GP2D
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import GPy
from operator import itemgetter
import sncosmo
import seaborn as sns
from tqdm import tqdm


class GP5D(GP2D):
    """The Gaussian Process for KNe Light Curves using GPy. This class handles the five dimensional case.
    """

    def __init__(self, referenceName):
        """ Instantiates class of both Gaussian Process and KNe Light Curve
        >>> gp = GP5D("reference.csv")
        >>> isinstance(gp, GP)
        True
        >>> isinstance(gp, LightCurve)
        True
        >>> isinstance(gp, float)
        False
        >>> isinstance(gp, GP2D)
        True
        >>> isinstance(gp, GP5D)
        True
        """
        GP2D.__init__(self, referenceName)
        self.num_wv = None
        self.training_shape = None
        self.Ntime = [0, 20, 100]  # THIS HAS BEEN FILLED IN MANUALLY
        self.num_pca_components = (4, 4)
        self.iobs_range = np.arange(0, 11, 1)  # THIS HAS BEEN FILLED IN MANUALLY
        self.isLog = True
        self.validationX = None
        self.cross_validation = None
        return None

    def set_wv_range(self, wv_range):
        """ Sets the wavelength range which save_training_data() uses.
        >>> data = GP5D("reference.csv")
        >>> data.set_wv_range(np.arange(600, 2600, 1000))
        >>> data.num_wv
        2
        >>> data.wv_range
        array([ 600, 1600])
        """
        self.num_wv = len(wv_range)
        self.wv_range = wv_range
        return None

    def save_training_data(self, time_trauncation=None, verbose=3):
        """ Saves the light curve data at specific wavelengths.
        >>> data = GP5D("reference.csv")
        >>> data.set_wv_range(np.arange(600, 2600, 1000))
        """
        print("[STATUS] Saving Training Data...")

        wv_range = self.wv_range
        self.time_shape = 100

        #         for index, row in self.reference.iterrows():
        def parallel_save_training(self, index, row, time_trauncation):
            phi_range = [row.phi]
            mejwind_range = [row.mejwind]
            mejdyn_range = [row.mejdyn]

            data = LightCurve("reference.csv")
            data.select_viewingangle(phi_range, mejdyn_range, mejwind_range, 900)

            for viewing_angle in self.iobs_range:
                save_matrix_list = []

                for i in wv_range:
                    data.select_viewingangle(phi_range, mejdyn_range, mejwind_range, i)

                    if time_trauncation is not None:
                        data.viewingangle = data.viewingangle.head(time_trauncation)
                        self.time_shape = time_trauncation

                    del data.viewingangle['time']

                    if data.viewingangle.shape == (100, 1) and viewing_angle != 0:
                        curr_matrix = data.viewingangle[0].to_numpy(dtype=float)
                    else:
                        curr_matrix = data.viewingangle[viewing_angle].to_numpy(dtype=float)

                    save_matrix_list.append(curr_matrix)

                    save_matrix = np.stack(save_matrix_list, axis=1)

                    epsilon = 1e-18
                    save_matrix[save_matrix == 0] = epsilon

                save_matrix = np.log10(save_matrix)
                np.save(
                    f"data/pca/mejdyn{mejdyn_range[0]}_mejwind{mejwind_range[0]}_phi{phi_range[0]}_iobs{viewing_angle}",
                    save_matrix)
            return None

        none_returned = Parallel(n_jobs=-2, verbose=verbose) \
            (delayed(parallel_save_training)(self, index, row, time_trauncation) \
             for index, row in self.reference.iterrows())

        return None

    def old_save_pca_components(self, skip_factor=None):
        """ Converts light curve training data into PCA components to be trained on in the emulator.
        """
        print("[STATUS] Saving PCA Components...")
        self.p1 = self.num_pca_components[0]
        self.p2 = self.num_pca_components[1]
        p1 = self.p1
        p2 = self.p2

        if skip_factor is not None:
            skip_factor -= 1

        self.skip_factor = skip_factor
        skip_counter = 0

        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:

                if skip_factor is not None:
                    skip_counter += 1

                    if skip_counter < skip_factor:
                        continue

                pca_matrix = np.load(
                    f"data/pca/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
                scaler = StandardScaler()
                scaler.fit(pca_matrix)
                scaled_data = scaler.transform(pca_matrix)
                pca = PCA(n_components=p1)
                pca.fit(scaled_data)
                X = pca.transform(scaled_data)
                pca2 = PCA(n_components=p2)
                pca2.fit(pca.components_.T)
                X2 = pca2.transform(pca.components_.T)
                training = pca2.components_.flatten()
                np.save(
                    f"data/pcaComponents/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy",
                    training)
                skip_counter = -1
        return None

    def save_pca_components(self, skip_factor=None):
        n_comp = self.n_comp
        lstX = []
        lstY = []

        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:
                if self.cross_validation is not None:
                    if (row.mejdyn == self.cross_validation[0]) and (row.mejwind == self.cross_validation[1]) and \
                            (row.phi == self.cross_validation[2]) and (viewing_angle == self.cross_validation[3]):
                        self.validationX = [row.mejdyn, row.mejwind, row.phi, viewing_angle]
                        continue
                pca_matrix = np.load(
                    f"data/pca/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
                lstX.append((row.mejdyn, row.mejwind, row.phi, viewing_angle))
                lstY.append(pca_matrix.flatten())

        aX = np.array(lstX)
        aY = np.array(lstY)
        aY = aY.T
        scaler = StandardScaler()
        scaler.fit(aY)
        scaled_data = scaler.transform(aY)
        pca = PCA(n_components=n_comp, svd_solver='randomized')
        pca.fit(scaled_data)
        X = pca.transform(scaled_data)
        counter = 0

        if skip_factor is not None:
            skip_factor -= 1

        self.skip_factor = skip_factor
        skip_counter = 0

        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:
                if (row.mejdyn == self.cross_validation[0]) and (row.mejwind == self.cross_validation[1]) and \
                        (row.phi == self.cross_validation[2]) and (viewing_angle == self.cross_validation[3]):
                    continue

                if skip_factor is not None:
                    skip_counter += 1
                    if skip_counter < skip_factor:
                        continue

                np.save(
                    f"data/pcaComponents/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy",
                    pca.components_[:, counter])
                skip_counter = -1
                counter += 1
        return None

    def setXY(self):
        """ Breaks components down into X and Y for the gaussian process.
        """
        print("[STATUS] Setting X, Y components for 5D Model.")
        x = []
        yList = []

        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:

                try:
                    y = np.load(
                        f"data/pcaComponents/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")

                except:
                    continue

                x.append([row.mejdyn, row.mejwind, row.phi, viewing_angle])
                tempY = y.flatten()
                assert np.allclose(y, tempY.reshape(y.shape))
                yList.append(tempY)

        self.training_shape = y.shape
        Y = np.array(yList, dtype=float)
        X = np.array(x, dtype=float)

        XNormed = X.copy()
        YNormed = Y.copy()

        for i in range(X.shape[1]):
            XNormed[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))

        self.unnormedX = X
        self.X = XNormed

        self.unnormedY = Y
        med = np.median(Y)
        self.median = med

        if np.isclose(self.median, 0):
            self.Y = Y
        else:
            self.Y = Y / med - 1
        return None

    def model_predict(self, predX=None, include_like=True):
        """ Predict new values of X and Y using optimized GP Emulator.
        """
        print("[STATUS] Predicting X and Y with trained emulator.")

        if predX is None:
            predX = self.X

        pred_arr, pred_var = self.model.predict(predX, include_likelihood=include_like)

        if not np.isclose(self.median, 0):
            pred_var = (pred_var + 1) * (self.median ** 2)
            pred_arr = (pred_arr + 1) * self.median

        self.predY = pred_arr

        pred_sigma = np.sqrt(pred_var)
        self.pred_sigma = pred_sigma.squeeze()

        for i in range(self.X.shape[0]):
            trained_pcaComponents = self.predY[i].reshape(self.training_shape)
            np.save(
                f"data/pcaComponentsTrained/mejdyn{self.unnormedX[i][0]}_mejwind{self.unnormedX[i][1]}_phi{int(self.unnormedX[i][2])}_iobs{int(self.unnormedX[i][3])}.npy",
                trained_pcaComponents)
            np.save(
                f"data/pcaComponentsTrainedError/mejdyn{self.unnormedX[i][0]}_mejwind{self.unnormedX[i][1]}_phi{int(self.unnormedX[i][2])}_iobs{int(self.unnormedX[i][3])}.npy",
                self.pred_sigma[i])

    def old_save_trained_data(self):
        """ Save new trained data in same format as original trained data.
        """
        p1 = self.p1
        p2 = self.p2
        print('[STATUS] Saving newly trained data predictions.')
        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:
                pca_matrix = np.load(
                    f"data/pca/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")

                try:
                    trained_pcaComponents = np.load(
                        f"data/pcaComponentsTrained/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
                except:
                    trained_pcaComponents = None
                    continue

                scaler = StandardScaler()
                scaler.fit(pca_matrix)
                scaled_data = scaler.transform(pca_matrix)
                pca = PCA(n_components=p1)
                pca.fit(scaled_data)
                X = pca.transform(scaled_data)
                pca2 = PCA(n_components=p2)
                pca2.fit(pca.components_.T)
                X2 = pca2.transform(pca.components_.T)
                pca2.components_ = trained_pcaComponents.reshape(self.p1, self.p2)
                trained2 = pca2.inverse_transform(pca2.transform(pca.components_.T)).T
                pca.components_ = trained2
                trained3 = pca.inverse_transform(pca.transform(scaled_data))
                inverted_trained_data = scaler.inverse_transform(trained3)

                assert inverted_trained_data.shape == pca_matrix.shape
                np.save(f"data/pcaTrained/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy",
                        inverted_trained_data)

    def save_trained_data_helper(self, typ=None):
        lstX = []
        lstY = []
        lstPCA = []
        n_comp = self.n_comp
        sigma = 1

        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:
                pca_matrix = np.load(
                    f"data/pca/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")

                try:
                    trainedComponents = np.load(
                        f"data/pcaComponentsTrained/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
                    trainedComponentsError = np.load(
                        f"data/pcaComponentsTrainedError/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")

                    if typ == "upper":
                        trainedComponents = trainedComponents + trainedComponentsError * sigma
                    if typ == "lower":
                        trainedComponents = trainedComponents - trainedComponentsError * sigma
                    if typ == None:
                        trainedComponents = trainedComponents

                except:
                    trainedComponents = None
                    continue

                lstPCA.append(trainedComponents)
                lstX.append((row.mejdyn, row.mejwind, row.phi, viewing_angle))
                lstY.append(pca_matrix.flatten())

        aX = np.array(lstX)
        aY = np.array(lstY)
        aPCA = np.array(lstPCA)
        aPCA = aPCA.T
        aY = aY.T
        scaler = StandardScaler()
        scaler.fit(aY)
        scaled_data = scaler.transform(aY)
        pca = PCA(n_components=n_comp, svd_solver='randomized')
        pca.fit(scaled_data)
        X = pca.transform(scaled_data)
        pca.components_ = aPCA
        trained1 = pca.inverse_transform(pca.transform(scaled_data))
        trained2 = scaler.inverse_transform(trained1)

        if self.skip_factor is not None:
            trained_pca_matrix = trained2.reshape(self.Ntime[2], self.num_wv,
                                                  int(np.floor(2156 / (self.skip_factor + 1))))
        else:
            trained_pca_matrix = trained2.reshape(self.Ntime[2], self.num_wv, 2156)

        counter = 0

        if self.cross_validation is None:
            for val in self.unnormedX:
                if typ == "upper":
                    np.save(f"data/pcaTrainedUpper/mejdyn{val[0]}_mejwind{val[1]}_phi{int(val[2])}_iobs{int(val[3])}.npy",
                            trained_pca_matrix[:, :, counter])
                if typ == "lower":
                    np.save(f"data/pcaTrainedLower/mejdyn{val[0]}_mejwind{val[1]}_phi{int(val[2])}_iobs{int(val[3])}.npy",
                            trained_pca_matrix[:, :, counter])
                if typ == None:
                    np.save(f"data/pcaTrained/mejdyn{val[0]}_mejwind{val[1]}_phi{int(val[2])}_iobs{int(val[3])}.npy",
                            trained_pca_matrix[:, :, counter])
                counter += 1

        if self.cross_validation is not None:
            if typ == "upper":
                np.save(f"data/pcaTrainedUpper/mejdyn{self.validationX[0]}_mejwind{self.validationX[1]}_phi{int(self.validationX[2])}_iobs{int(self.validationX[3])}.npy",
                        trained_pca_matrix[:, :, -1])
            if typ == "lower":
                np.save(f"data/pcaTrainedLower/mejdyn{self.validationX[0]}_mejwind{self.validationX[1]}_phi{int(self.validationX[2])}_iobs{int(self.validationX[3])}.npy",
                        trained_pca_matrix[:, :, -1])
            if typ is None:
                np.save(f"data/pcaTrained/mejdyn{self.validationX[0]}_mejwind{self.validationX[1]}_phi{int(self.validationX[2])}_iobs{int(self.validationX[3])}.npy",
                        trained_pca_matrix[:, :, -1])
        return None

    def save_trained_data(self):
        self.save_trained_data_helper(typ=None)
        self.save_trained_data_helper(typ="lower")
        self.save_trained_data_helper(typ="upper")
        return None

    def LOOCV_PCA(self, verbose=4):
        """ Leave one light curve out, and predict its flux and compare with what the true data says.
        """

        def parallel_LOOCV(self, i):
            tempX = self.X.copy()  # Normalized
            tempY = self.Y.copy()  # Normalized
            tempKernel = self.kernel.copy()

            test_pointX = np.array([tempX[i]])
            test_pointY = np.array(tempY[i])
            tempX = np.delete(tempX, i, 0)
            tempY = np.delete(tempY, i, 0)
            self.set_kernel(tempKernel.copy())
            self.model = GPy.models.GPRegression(tempX, tempY, self.kernel)
            #             self.model['.*lengthscale'].constrain_bounded(0,5)
            self.model.optimize(messages=False)

            predY, varY = self.model.predict(test_pointX)

            # Undo Normalization
            if not np.isclose(self.median, 0):
                varY = (varY + 1) * (self.median ** 2)
                predY = (predY + 1) * self.median
                test_pointY = (test_pointY + 1) * self.median

            # Update final LOO List
            looList = (test_pointY - predY) / np.sqrt(varY)
            #             lengthscale = list(self.model.rbf.lengthscale)
            lengthscale = list(self.model.mul.rbf.lengthscale)

            trained_pcaComponents = predY.reshape(self.training_shape)
            np.save(
                f"data/pcaComponentsTrained/mejdyn{self.unnormedX[i][0]}_mejwind{self.unnormedX[i][1]}_phi"
                f"{int(self.unnormedX[i][2])}_iobs{int(self.unnormedX[i][3])}.npy",
                trained_pcaComponents)

            return looList, lengthscale

        self.results = Parallel(n_jobs=-2, verbose=verbose) \
            (delayed(parallel_LOOCV)(self, i) for i in range(len(self.X)))
        self.list_lengthscale = np.array(list(map(itemgetter(1), self.results)), dtype=float)
        self.looListMultiple = np.array(list(map(itemgetter(0), self.results)), dtype=float).squeeze()
        self.loo_list_multiple = np.squeeze(self.looListMultiple)

        return None

    def ComputeDifferenceFlux(self):
        """ Final pipeline of LOOCV, computes the difference between the training and predictive light curve data.
        """
        self.empirical = True
        self.list_looList = []
        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:
                try:
                    untrained = np.load(
                        f"data/pca/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
                    trained = np.load(
                        f"data/pcaTrained/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
                except:
                    continue
                currLoo = (trained - untrained) / untrained
                self.list_looList.append(currLoo)

        self.difference = np.array(self.list_looList, dtype=float)
        return None

    def plot_difference_histogram(self, edge=2.5, mu=0, sigma=1, binning=30):
        """ Plots a histogram of the differences.
        """
        fig, ax = plt.subplots(dpi=300)
        utkarshGrid()

        hist_arr = self.difference.flatten()
        hist_arr = hist_arr[np.isfinite(hist_arr)]
        hist_arr = hist_arr[hist_arr < 3]
        hist_arr = hist_arr[hist_arr > -3]
        print(f"Inside 3x: {len(hist_arr)}, Total: {len(self.difference.flatten())}")

        df = pd.DataFrame(hist_arr, columns=["hist"])

        y, binEdges = np.histogram(hist_arr, bins=binning, density=True, normed=True)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        menStd = np.sqrt(y)
        mx = y + np.sqrt(y)
        mn = y - np.sqrt(y)
        plt.fill_between(bincenters, mn, mx, alpha=0.2, zorder=3, color="limegreen")

        if not self.empirical:
            x_gauss = np.linspace(-edge, edge, 100, endpoint=True)
            y_gauss = self.gaussian(x_gauss, mu, sigma)
            plt.plot(x_gauss, y_gauss, label="Unit Gaussian", color="purple", zorder=3)

        df.plot.hist(density=True, bins=binning, ax=ax, label="Count",
                     facecolor='#2ab0ff', edgecolor='#169acf', zorder=1)

        df.plot.kde(ax=ax, label="LOOCV Distribution", alpha=1, zorder=2, color="green")
        plt.ylabel("Count Intensity (Log Flux Difference)")
        plt.xlabel(r"Deviation Error (Units Log Flux)")

        if self.empirical:
            ax.legend(["Density Distribution ", "Count", r"1 $\sigma$ Confidence"])
            ax.set_title(r"Flux Ratio = $\frac{{Truth - Predictive}}{Truth}$")


        else:
            ax.legend(["Unit Gaussian", "Difference Distribution ", "Count"])
            ax.set_title(r"Flux Ratio = $\frac{{Truth - Predictive}}{\sigma}$")

        ax.set_ylim(bottom=-0.1)

    def plot_filters(self, mejdyn, mejwind, phi, iobs, colors="coolwarm_r"):
        t = np.arange(self.Ntime[0], self.Ntime[1], self.Ntime[1] / self.Ntime[2])
        untrained = np.load(f"data/pca/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        trained = np.load(f"data/pcaTrained/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        filters = ["sdss::u", "sdss::g", "sdss::r", "sdss::i", "sdss::z",
                   "swope2::y", "swope2::J", "swope2::H"]

        colors = sns.color_palette(colors, len(filters))[::-1]

        plt.figure(dpi=300, figsize=(6, 3))
        for i in range(len(filters)):
            source = sncosmo.TimeSeriesSource(t, self.wv_range * 10, 10 ** trained)
            source2 = sncosmo.TimeSeriesSource(t, self.wv_range * 10, 10 ** untrained)
            m = source.bandmag(filters[i], "ab", t)
            m2 = source2.bandmag(filters[i], "ab", t)
            plt.plot(t, m, label=f"{filters[i][-1]}", color=colors[i])
            plt.plot(t, m2, linestyle="dotted", alpha=0.3, color=colors[i])

        plt.legend()
        utkarshGrid()
        plt.gca().invert_yaxis()
        plt.xlabel("Time (days)")
        plt.ylabel("Flux (Magnitude)")
        plt.title(f"Magnitude plot as specified filters")
        return None

    def get_flux(self, mejdyn, mejwind, phi, iobs, time_desired, wv_desired):
        t = np.arange(self.Ntime[0], self.Ntime[1], self.Ntime[1] / self.Ntime[2])
        wv_index = (np.abs(self.wv_range - wv_desired)).argmin()  # Plot arbitary wavelength.
        time_index = (np.abs(t - time_desired)).argmin()  # Plot arbitary wavelength.
        trained = np.load(f"data/pcaTrained/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")

        print(f"=== Flux Estimation === \nmejdyn: {mejdyn}\nmejwind: {mejwind}\
        \nphi: {phi}\nviewing_angle: {iobs}\nwavelength: {self.wv_range[wv_index]}nm\
        \ntime: {round(t[time_index], 2)} days\n\nLOG FLUX: {round(trained[time_index, wv_index], 5)}")
        return None

    def overplot_time(self, mejdyn, mejwind, phi, iobs, wv_desired):
        t = np.arange(self.Ntime[0], self.Ntime[1], self.Ntime[1] / self.Ntime[2])
        wv_index = (np.abs(self.wv_range - wv_desired)).argmin()  # Plot arbitary wavelength.
        trained = np.load(f"data/pcaTrained/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        trainedUpper = np.load(f"data/pcaTrainedUpper/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        trainedLower = np.load(f"data/pcaTrainedLower/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        untrained = np.load(f"data/pca/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")

        plt.figure(dpi=300, figsize=(6, 3))
        plt.title(f"Wavelength = {self.wv_range[wv_index]}nm")
        plt.plot(t, untrained[:, wv_index], label="Training Data", color="purple")
        # plt.plot(t, trainedUpper[:, wv_index], alpha=0.3, color="lightblue", label=r"1$\sigma$")
        # plt.plot(t, trainedLower[:, wv_index], alpha=0.3, color="lightblue")
        plt.plot(t, trained[:, wv_index], label="Trained Emulator + PCA", linestyle="dashed", color="dodgerblue")
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Log Flux (Magnitude)")
        utkarshGrid()
        return None

    def overplot_wavelength(self, mejdyn, mejwind, phi, iobs, time_desired):
        t = np.arange(self.Ntime[0], self.Ntime[1], self.Ntime[1] / self.Ntime[2])
        time_index = (np.abs(t - time_desired)).argmin()  # Plot arbitary wavelength.
        trained = np.load(f"data/pcaTrained/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        trainedUpper = np.load(f"data/pcaTrainedUpper/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        trainedLower = np.load(f"data/pcaTrainedLower/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
        untrained = np.load(f"data/pca/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")

        plt.figure(dpi=300, figsize=(6, 3))
        plt.title(f"Time = {round(t[time_index], 3)} Days")
        plt.plot(self.wv_range, untrained[time_index, :], label="Training Data", color="purple")
        plt.plot(self.wv_range, trained[time_index, :], label="Trained Emulator + PCA",
                 linestyle="dashed", color="dodgerblue")
        # plt.plot(self.wv_range, trainedUpper[time_index, :], alpha=0.3, color="lightblue",
        #          label=r"1$\sigma$ Error (UNFINISHED)")
        # plt.plot(self.wv_range, trainedLower[time_index, :], alpha=0.3, color="lightblue")
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Log Flux (Magnitude)")
        utkarshGrid()
        return None

    def plot_emulator_errors(self):
        t = np.arange(self.Ntime[0], self.Ntime[1], self.Ntime[1] / self.Ntime[2])
        diff = np.zeros(t.shape)
        for index, row, in tqdm(self.reference.iterrows(), total=196):
            for viewing_angle in self.iobs_range:
                mejdyn = row.mejdyn
                mejwind = row.mejwind
                phi = row.phi
                iobs = viewing_angle

                untrained = np.load(f"data/pca/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")
                trained = np.load(f"data/pcaTrained/mejdyn{mejdyn}_mejwind{mejwind}_phi{phi}_iobs{iobs}.npy")

                for i in range(len(t)):
                    diff += abs(trained[:, i] - untrained[:, i])

        plt.figure(dpi=300, figsize=(6, 3))
        plt.plot(t, diff, color="goldenrod")
        utkarshGrid()
        plt.title("Errors between Bulla model and the emulator")
        # plt.ylabel("Absolute Error (Cumulative Magnitude)")
        plt.ylabel("Absolute Error (Cumulative Log Flux)")
        plt.xlabel("Time (days)")
        plt.show()

    def setXY_cross_validation(self, mejdyn, mejwind, phi, iobs):
        print("[STATUS] Setting X, Y components for 5D Model.")
        x = []
        yList = []

        for index, row, in self.reference.iterrows():
            for viewing_angle in self.iobs_range:
                if (row.mejdyn == mejdyn) and (row.mejwind == mejwind) and (row.phi == phi) and (viewing_angle == iobs):
                    self.validationX = [row.mejdyn, row.mejwind, row.phi, viewing_angle]
                    continue
                else:
                    pass

                try:
                    y = np.load(
                        f"data/pcaComponents/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")

                except:
                    continue

                x.append([row.mejdyn, row.mejwind, row.phi, viewing_angle])
                tempY = y.flatten()
                assert np.allclose(y, tempY.reshape(y.shape))
                yList.append(tempY)

        self.training_shape = y.shape
        Y = np.array(yList, dtype=float)
        X = np.array(x, dtype=float)

        XNormed = X.copy()
        x.append(self.validationX)
        XValidation = np.array(x, dtype=float)
        XNormedValidation = XValidation.copy()
        YNormed = Y.copy()

        for i in range(X.shape[1]):
            XNormed[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
            XNormedValidation[:, i] = (XValidation[:, i] - min(XValidation[:, i])) /\
                                      (max(XValidation[:, i]) - min(XValidation[:, i]))

        self.validationXNormed = XNormedValidation[-1]
        self.unnormedX = X
        self.X = XNormed

        self.unnormedY = Y
        med = np.median(Y)
        self.median = med

        if np.isclose(self.median, 0):
            self.Y = Y
        else:
            self.Y = Y / med - 1

        assert len(self.Y) == 196 * 11 - 1
        return None

    def model_predict_cross_validation(self, include_like=True):
        """ Predict new values of X and Y using optimized GP Emulator.
        """
        print("[STATUS] Predicting X and Y with trained emulator.")
        predX = self.validationXNormed.reshape(1,len(self.validationXNormed))

        pred_arr, pred_var = self.model.predict(predX, include_likelihood=include_like)

        if not np.isclose(self.median, 0):
            pred_var = (pred_var + 1) * (self.median ** 2)
            pred_arr = (pred_arr + 1) * self.median

        self.predY = pred_arr

        pred_sigma = np.sqrt(pred_var)
        self.pred_sigma = pred_sigma.squeeze()

        trained_pcaComponents = self.predY.reshape(self.training_shape)
        np.save(f"data/pcaComponentsTrained/mejdyn{self.validationX[0]}_mejwind{self.validationX[1]}_phi{int(self.validationX[2])}_iobs{int(self.validationX[3])}.npy",
                trained_pcaComponents)
        np.save(f"data/pcaComponentsTrainedError/mejdyn{self.validationX[0]}_mejwind{self.validationX[1]}_phi{int(self.validationX[2])}_iobs{int(self.validationX[3])}.npy",
                self.pred_sigma)
        return None

    # def save_trained_data_cross_validation(self, mejdyn, mejwind, phi, iobs, typ=None):
    #     lstX = []
    #     lstY = []
    #     lstPCA = []
    #     n_comp = self.n_comp
    #     sigma = 1
    #
    #     for index, row, in self.reference.iterrows():
    #         for viewing_angle in self.iobs_range:
    #             if (row.mejdyn == mejdyn) and (row.mejwind == mejwind) and (row.phi == phi) and (viewing_angle == iobs):
    #                 self.validationX = [row.mejdyn, row.mejwind, row.phi, viewing_angle]
    #                 validation_matrix = np.load(
    #                     f"data/pca/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
    #                 validation_components = np.load(
    #                     f"data/pcaComponentsTrained/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}")
    #                 validation_components_errpr = np.load(
    #                     f"data/pcaComponentsTrainedError/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
    #                 continue
    #             else:
    #                 pass
    #             pca_matrix = np.load(
    #                 f"data/pca/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
    #
    #             try:
    #                 trainedComponents = np.load(
    #                     f"data/pcaComponentsTrained/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
    #                 trainedComponentsError = np.load(
    #                     f"data/pcaComponentsTrainedError/mejdyn{row.mejdyn}_mejwind{row.mejwind}_phi{row.phi}_iobs{viewing_angle}.npy")
    #
    #                 if typ == "upper":
    #                     trainedComponents = trainedComponents + trainedComponentsError * sigma
    #                 if typ == "lower":
    #                     trainedComponents = trainedComponents - trainedComponentsError * sigma
    #                 if typ == None:
    #                     trainedComponents = trainedComponents
    #
    #             except:
    #                 trainedComponents = None
    #                 continue
    #
    #             lstPCA.append(trainedComponents)
    #             lstX.append((row.mejdyn, row.mejwind, row.phi, viewing_angle))
    #             lstY.append(pca_matrix.flatten())
    #
    #     aX = np.array(lstX)
    #     aY = np.array(lstY)
    #     aPCA = np.array(lstPCA)
    #     aPCA = aPCA.T
    #     aY = aY.T
    #     scaler = StandardScaler()
    #     scaler.fit(aY)
    #     scaled_data = scaler.transform(aY)
    #     pca = PCA(n_components=n_comp, svd_solver='randomized')
    #     pca.fit(scaled_data)
    #     X = pca.transform(scaled_data)
    #     pca.components_ = aPCA
    #     trained1 = pca.inverse_transform(pca.transform(scaled_data))
    #     trained2 = scaler.inverse_transform(trained1)
    #
    #     if self.skip_factor is not None:
    #         trained_pca_matrix = trained2.reshape(self.Ntime[2], self.num_wv,
    #                                               int(np.floor(2155 / (self.skip_factor + 1))))
    #     else:
    #         trained_pca_matrix = trained2.reshape(self.Ntime[2], self.num_wv, 2155)
    #
    #     counter = 0
    #
    #     print(trained_pca_matrix.shape)
    #
    #     for val in self.unnormedX:
    #         if typ == "upper":
    #             np.save(f"data/pcaTrainedUpper/mejdyn{val[0]}_mejwind{val[1]}_phi{int(val[2])}_iobs{int(val[3])}.npy",
    #                     trained_pca_matrix[:, :, counter])
    #         if typ == "lower":
    #             np.save(f"data/pcaTrainedLower/mejdyn{val[0]}_mejwind{val[1]}_phi{int(val[2])}_iobs{int(val[3])}.npy",
    #                     trained_pca_matrix[:, :, counter])
    #         if typ == None:
    #             np.save(f"data/pcaTrained/mejdyn{val[0]}_mejwind{val[1]}_phi{int(val[2])}_iobs{int(val[3])}.npy",
    #                     trained_pca_matrix[:, :, counter])
    #         counter += 1
    #     return None