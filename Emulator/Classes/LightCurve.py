import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm


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


class LightCurve():
    """ The information regarding KNe light curves and data corresponding to KNe light curves.
    """

    def __init__(self, referenceName):
        """ Initializes class, reference is all the light curves, and selected represents ones of interest to be narrowed.
        """
        self.reference = pd.read_csv(referenceName)
        self.selected = self.reference.copy()
        self.uBand = 365
        self.bBand = 445
        self.gBand = 464
        self.vBand = 551
        self.rBand = 658
        self.iBand = 806
        self.zBand = 900
        self.yBand = 1020
        self.jBand = 1220
        self.hBand = 1630
        self.kBand = 2190
        self.lBand = 3450
        self.mBand = 4750
        self.nBand = 10500
        self.qBand = 21000
        self.temp_path = "/Users/utkarsh/PycharmProjects/SURP2021"

        warnings.filterwarnings(action='ignore', module='matplotlib.figure', category=UserWarning,
                                message=('This figure includes Axes that are not compatible with tight_layout, '
                                         'so results might be incorrect.'))

    def _slice(self, typ, Min, Max):
        sliced = self.selected[self.selected[typ] >= Min]
        sliced2 = sliced[sliced[typ] <= Max]
        return sliced2

    def select_curve(self, phiRange=[], mejdynRange=[], mejwindRange=[], nphRange=[]):
        """ Select a measurment based on the physics limits required.
        >>> data = LightCurve("reference.csv")
        >>> phi_range = [30]
        >>> mejdyn_range = [0.01]
        >>> mejwind_range = [0.11]
        >>> data.select_curve(phiRange = phi_range, mejdynRange = mejdyn_range, mejwindRange = mejwind_range)
        >>> print(data.selected.filename.iloc[0])
        nph1.0e+06_mejdyn0.010_mejwind0.110_phi30.txt
        """
        self.phi_range_single = phiRange
        self.mejdyn_range_single = mejdynRange
        self.mejwind_range_single = mejwindRange
        self.nph_range_single = nphRange
        if len(nphRange) > 0:
            self.selected = self._slice("nph", min(nphRange), max(nphRange))
        if len(phiRange) > 0:
            self.selected = self._slice("phi", min(phiRange), max(phiRange))
        if len(mejdynRange) > 0:
            self.selected = self._slice("mejdyn", min(mejdynRange), max(mejdynRange))
        if len(mejwindRange) > 0:
            self.selected = self._slice("mejwind", min(mejwindRange), max(mejwindRange))
        return None

    def _set_path(self, path):
        """ Sets the path to the file to be extracted. Chooses first file if there are many.
        >>> data = LightCurve("reference.csv")
        >>> data._set_path("")
        [WARNING] Many curves in data: First curve has been selected.
        [CURVE] nph1.0e+06_mejdyn0.001_mejwind0.130_phi45.txt
        >>> print(data.path)
        /bns_m3_3comp/nph1.0e+06_mejdyn0.001_mejwind0.130_phi45.txt
        """
        self.temp_path = path
        self.folder_path = path + "/bns_m3_3comp/"
        self.path = self.folder_path + self.selected.filename.iloc[0]
        if len(self.selected.filename) > 1:
            print(
                f"[WARNING] Many curves in data: First curve has been selected."
                f"\n[CURVE] {self.selected.filename.iloc[0]}")

        return None

    def extract_curve(self):
        """ Extracts curve based on selected data and converts it into a readable format.
        >>> data = LightCurve("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.select_curve(phiRange = phi_range, mejdynRange = mejdyn_range, mejwindRange = mejwind_range)
        >>> data.extract_curve()
        >>> data.curve.shape
        (11, 500)
        >>> zBand = 910
        >>> plotDf = data.curve.loc[:, [zBand]]
        >>> print(plotDf.loc[1,zBand][3])
        0.0028678
        >>> print(data.selected.filename.iloc[0])
        nph1.0e+06_mejdyn0.020_mejwind0.110_phi60.txt
        >>> data.curve.shape
        (11, 500)
        >>> data.Nobs
        11
        >>> data.Nwave
        500.0
        >>> data.Ntime
        [100.0, 0.0, 20.0]
        >>> data.time_arr[13]
        2.6262626262626263
        >>> data = LightCurve("reference.csv")
        >>> phi_range = [61]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.select_curve(phiRange = phi_range, mejdynRange = mejdyn_range, mejwindRange = mejwind_range)
        >>> data.extract_curve()
        [ERROR] Selected dataframe is empty! Data Compromised.
        """

        if self.selected.empty:
            print('[ERROR] Selected dataframe is empty! Data Compromised.')
            return None

        # Obtain path to read curve from.
        self._set_path(self.temp_path)

        # Read txt file containig light curve information
        temp0 = pd.read_csv(self.path, header=None, names=["data"])

        # Set parameters for viewing angles, numbers of wavelengths, and time step.
        self.Nobs = int(temp0.data.iloc[0])
        self.Nwave = float(temp0.data.iloc[1])
        self.Ntime = list(map(float, temp0.data.iloc[2].split()))
        self.time_arr = np.linspace(int(self.Ntime[1]), int(self.Ntime[2]), int(self.Ntime[0]), endpoint=True)

        # Drop information header and reset index.
        temp1 = temp0.iloc[3:].reset_index(drop=True)

        # Convert data from string to float
        temp1["data"] = temp1["data"].apply(lambda x: list(map(float, x.split())))

        # Obtain wavelength from messy data list. Convert to nm
        temp1.loc[:, 'wavelength'] = temp1.data.map(lambda x: x[0] / 10)

        # Remove wavelengths from data vector.
        temp1["data"] = temp1["data"].apply(lambda x: x[1:])

        # Pivot to order the table by wavelengths
        temp1 = temp1.pivot(columns="wavelength", values="data")

        # Concatenate all rows to remove NA values to get a neat, readable dataframe.
        final = pd.concat([temp1[col].dropna().reset_index(drop=True) for col in temp1], axis=1)

        # Rename axis titles.
        final.index.name = "iobs"
        final.columns.name = "wavelength"
        self.curve = final

        return None

    def _odd(self, x):
        """Rounds to nearest odd numbers
        >>> data = LightCurve("reference.csv")
        >>> data._odd(3)
        3
        >>> data._odd(2.5)
        3
        >>> data._odd(2)
        3
        >>> data._odd(1.999)
        1
        """
        return 2 * int(x / 2) + 1

    def simple_plot(self, wv):
        """Simple plotting function by wavelength for light curve data.
        >>> data = LightCurve("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.select_curve(phiRange = phi_range, mejdynRange = mejdyn_range, mejwindRange = mejwind_range)
        >>> data.extract_curve()
        >>> data.simple_plot(900)
        [STATUS] Plotting...
        >>> data.time_arr[11]
        2.2222222222222223
        >>> data.wavelength
        910
        >>> plt.show() #doctest: +SKIP
        >>> plt.close()
        """
        print("[STATUS] Plotting...")
        self.time_arr = np.linspace(int(self.Ntime[1]), int(self.Ntime[2]), int(self.Ntime[0]), endpoint=True)
        self.wavelength = 10 * self._odd(wv / 10)

        viewing_angles = np.linspace(0, 1, self.Nobs, endpoint=True)
        plt.figure(dpi=300)
        plt.gca().set_prop_cycle("color", sns.color_palette("coolwarm_r", self.Nobs))
        for i, j in self.curve.loc[:, [self.wavelength]].iterrows():
            ang = round(np.degrees(np.arccos(viewing_angles[i])), 2)
            plt.plot(self.time_arr, j.values[0], label=f"{ang}"r"$^o$", linewidth=1)
        plt.xlabel("Time (Days)")
        plt.ylabel(r"Flux $Erg s^{-1} cm^{-2}A^{-1}$")
        plt.title(f"Lights curves for {self.Nobs} viewing angles at {self.wavelength}nm")
        utkarshGrid()
        plt.legend(title=r"$\Phi$")
        return None

    def _compute_wavelength(self, wv):
        """Wavelength helper function. Rounds value to nearest odd 10.
        >>> data = LightCurve("reference.csv")
        >>> data._compute_wavelength(900)
        910
        >>> data._compute_wavelength(899.99)
        890
        """
        return 10 * self._odd(wv / 10)

    def plot_viewingangle_simple(self):
        """Plots LightCurve according to viewing angle from the pole to the equator. Multiple bands are plotted.
        >>> data = LightCurve("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.select_curve(phiRange = phi_range, mejdynRange = mejdyn_range, mejwindRange = mejwind_range)
        >>> data.extract_curve()
        >>> data.plot_viewingangle_simple()
        [STATUS] Plotting...
        >>> plt.show() #doctest: +SKIP
        >>> plt.close()
        """
        wvList = [self.uBand, self.gBand, self.rBand, self.iBand,
                  self.zBand, self.yBand, self.jBand, self.hBand]

        plt.figure(dpi=300)
        print("[STATUS] Plotting...")
        for k in range(len(wvList)):
            wv = wvList[k]
            self.time_arr = np.linspace(int(self.Ntime[1]), int(self.Ntime[2]), int(self.Ntime[0]), endpoint=True)
            self.wavelength = self._compute_wavelength(wv)
            viewing_angles = np.linspace(0, 1, self.Nobs, endpoint=True)
            colors = sns.color_palette("coolwarm_r", len(wvList))[::-1]
            # plt.gca().set_prop_cycle("color", sns.color_palette("coolwarm_r",len(wvList)))

            for i, j in self.curve.loc[:, [self.wavelength]].iterrows():
                if i == 0:
                    labelStr = f"{self.wavelength}nm"
                else:
                    labelStr = f""
                ang = round(np.degrees(np.arccos(viewing_angles[i])), 2)
                plt.plot(self.time_arr, j.values[0], label=labelStr,
                         linewidth=1, color=colors[k])

        plt.xlabel("Time (Days)")
        plt.ylabel(r"Log Flux $Erg s^{-1} cm^{-2}A^{-1}$")
        plt.legend(title=r"$\lambda$", ncol=2, loc="upper right")
        plt.yscale("log")
        plt.title(f"Lights curves for {self.Nobs} viewing angles at varying wavelengths")
        return None

    def plot_viewingangle(self):
        """ Plots LightCurve according to viewing angle as mutiple subplots.
        >>> data = LightCurve("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.select_curve(phiRange = phi_range, mejdynRange = mejdyn_range, mejwindRange = mejwind_range)
        >>> data.extract_curve()
        >>> data.plot_viewingangle()
        [STATUS] Plotting for nph: [], mejdyn: [0.02], mejwind: [0.11], phi: [60], viewing_angle: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
        >>> data.uBand
        365
        >>> data.counter_final
        8
        >>> plt.show() #doctest: +SKIP
        >>> plt.close()
        """
        fig, axes = plt.subplots(nrows=2, ncols=4, dpi=300, figsize=(6, 3.5))
        plt.tight_layout()
        viewing_angles = np.linspace(0, 1, self.Nobs, endpoint=True)
        colors = plt.cm.RdBu(np.linspace(-1, 1, self.Nobs))
        counter = 0
        wvList = [self.uBand, self.gBand, self.rBand, self.iBand,
                  self.zBand, self.yBand, self.jBand, self.hBand]
        namesList = ["uBand", "gBand", "rBand", "iBand", "zBand", "yBand", "jBand", "hBand"]
        ticks = np.arange(min(self.time_arr), max(self.time_arr) + 1, 5)
        self.iobs_range = viewing_angles

        for row in range(0, 2):
            for col in range(0, 4):
                wv = wvList[counter]
                self.wavelength = self._compute_wavelength(wv)

                for i, j in self.curve.loc[:, [self.wavelength]].iterrows():
                    if i == 0:

                        labelStr = f"{namesList[counter][0].upper()}"
                    else:
                        labelStr = f""
                    ang = round(np.degrees(np.arccos(viewing_angles[i])), 2)
                    im = axes[row, col].plot(self.time_arr, j.values[0], label=labelStr,
                                             linewidth=1, color=colors[i])

                axes[row, col].set_yscale('log')
                axes[row, col].legend(handletextpad=-2.0, handlelength=0)
                axes[row, col].axes.get_yaxis().set_visible(False)
                axes[row, col].set_xticks(ticks)

                counter += 1

        self.counter_final = counter
        bottom, top = 0.1, 0.9
        left, right = 0.2, 0.8

        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        my_cmap = "RdBu"
        sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=90))
        cbar_ax = fig.add_axes([1, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), extend='both', cax=cbar_ax,
                            format=mpl.ticker.FuncFormatter(self._fmt_degree))
        cbar.set_label(r"Viewing Angle $\theta_0$", size=15, labelpad=10)
        fig.text(0.5, 0, "Time since merger (Days)", ha='center')
        fig.text(-0.01, 0.5, r"Log Flux $Erg s^{-1} cm^{-2}A^{-1}$", va='center', rotation='vertical')
        print(
            f"[STATUS] Plotting for nph: {self.nph_range_single}, mejdyn: {self.mejdyn_range_single}, mejwind: {self.mejwind_range_single}, phi: {self.phi_range_single}, viewing_angle: {self.iobs_range}")
        return None

    def select_viewingangle(self, phi_range, mejdyn_range, mejwind_range, wv=0):
        """ Trauncates the selected data (self.curve) by selected wavelength.
        >>> data = LightCurve("reference.csv")
        >>> phi_range = [60]
        >>> mejdyn_range = [0.02]
        >>> mejwind_range = [0.11]
        >>> data.select_curve(phiRange = phi_range, mejdynRange = mejdyn_range, mejwindRange = mejwind_range)
        >>> data.extract_curve()
        >>> data.select_viewingangle(phi_range, mejdyn_range, mejwind_range, 900)
        >>> data.viewingangle.iloc[0][3]
        2.6453e-06
        >>> data.viewingangle.iloc[3][0]
        0.0019884
        >>> data.viewingangle.iloc[3][3]
        0.0029003
        >>> data.viewingangle.shape
        (100, 12)
        >>> data.mejdyn_range
        [0.02]
        >>> data.wv_range
        900
        >>> data.phi_range
        [60]
        >>> data.mejwind_range
        [0.11]
        """
        self.mejdyn_range = mejdyn_range
        self.wv_range = wv
        self.phi_range = phi_range
        self.mejwind_range = mejwind_range

        wv = self._compute_wavelength(wv)
        self.select_curve(phiRange=phi_range,
                          mejdynRange=mejdyn_range,
                          mejwindRange=mejwind_range)
        self.extract_curve()

        if wv > 0:
            z = self.curve.T[self.curve.T.index == wv]
            z = z.reset_index(drop=True)
            z = z.apply(pd.Series.explode).reset_index(drop=True)
            z["time"] = self.time_arr
            z.index.name = "time_step"
            self.viewingangle = z
        return None

    def select_mejdyn(self, wv_range, iobs_range, phi_range, mejwind_range):
        """Slice reference data set by dynamical ejecta mass (mejdyn).
        >>> data = LightCurve("reference.csv")
        >>> wv_range = [900]
        >>> iobs_range = [0]
        >>> phi_range = [45]
        >>> mejwind_range = [0.13]
        >>> data.select_mejdyn(wv_range, iobs_range, phi_range, mejwind_range)
        >>> data.mejdyn.iloc[0,3]
        2.8034e-06
        >>> data.mejdyn.iloc[44,2]
        0.00019771
        >>> data.mejdyn.iloc[65,3]
        4.5352e-05
        >>> data.mejdyn.shape
        (100, 4)
        >>> data.phi_range
        [45]
        >>> data.mejdyn_range
        [0.001, 0.005, 0.01, 0.02]
        >>> data.iobs_range
        [0]
        """
        mejdyn_range_list = self.reference.mejdyn.unique()

        self.mejdyn_range = sorted(mejdyn_range_list)
        self.wv_range = wv_range
        self.iobs_range = iobs_range
        self.phi_range = phi_range
        self.mejwind_range = mejwind_range

        mejdyn_range_list = [[x] for x in mejdyn_range_list]
        df = pd.DataFrame()
        self.select_curve(phiRange=phi_range, mejwindRange=mejwind_range)
        tempReference = self.selected

        for i in range(len(mejdyn_range_list)):
            self.select_curve(mejdynRange=[self.mejdyn_range[i]])
            self.extract_curve()
            arr = self.curve[self._compute_wavelength(wv_range[0])][
                iobs_range[0]]  # at wv 900, iobs0, over TIME STEP, for mejdyn 0.01
            assert len(arr) == self.Ntime[0]
            df[self.mejdyn_range[i]] = arr
            self.selected = tempReference

        df.columns.name = "mejdyn"
        df.index.name = "time_step"
        self.mejdyn = df
        self.mejdyn = self.mejdyn.sort_index(axis=1)
        return None

    def plot_mejdyn(self, verbose=False):
        """ Plots the data acquired in the dynamical ejecta mass.
        >>> data = LightCurve("reference.csv")
        >>> wv_range = [900]
        >>> iobs_range = [0]
        >>> phi_range = [45]
        >>> mejwind_range = [0.13]
        >>> data.select_mejdyn(wv_range, iobs_range, phi_range, mejwind_range)
        >>> data.plot_mejdyn() #doctest: +SKIP
        """
        numRows = 2
        numCols = 4
        fig, axes = plt.subplots(nrows=numRows, ncols=numCols, dpi=300, figsize=(6, 3.5))
        plt.tight_layout()
        viewing_angles = np.linspace(0, 1, self.Nobs, endpoint=True)
        colors = plt.cm.PiYG(np.linspace(0, 1, len(self.mejdyn_range)))
        counter = 0
        namesList = list(map(str, self.mejdyn_range))
        ticks = np.arange(min(self.time_arr), max(self.time_arr) + 1, 5)
        wvList = [self.uBand, self.gBand, self.rBand, self.iBand,
                  self.zBand, self.yBand, self.jBand, self.hBand]
        namesList = ["uBand", "gBand", "rBand", "iBand", "zBand", "yBand", "jBand", "hBand"]

        row = 0
        col = 0

        for row in tqdm(range(0, numRows), disable=not verbose):
            for col in range(0, numCols):
                wv = wvList[counter]
                self.select_mejdyn([wv], self.iobs_range, self.phi_range, self.mejwind_range)
                self.wavelength = self._compute_wavelength(wv)

                for i in range(len(self.mejdyn_range)):
                    if i == 0:

                        labelStr = f"{namesList[counter][0].upper()}"
                    else:
                        labelStr = f""
                    axes[row, col].plot(self.time_arr, self.mejdyn[self.mejdyn_range[i]], label=labelStr,
                                        linewidth=1, color=colors[i])
                axes[row, col].set_yscale('log')
                axes[row, col].legend(handletextpad=-2.0, handlelength=0)
                axes[row, col].axes.get_yaxis().set_visible(False)
                axes[row, col].set_xticks(ticks)

                counter += 1

        self.counter_final = counter
        bottom, top = 0.1, 0.9
        left, right = 0.2, 0.8

        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        my_cmap = "PiYG"
        sm = plt.cm.ScalarMappable(cmap=my_cmap,
                                   norm=plt.Normalize(vmin=min(self.mejdyn_range), vmax=max(self.mejdyn_range)))
        cbar_ax = fig.add_axes([1, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), extend='both', cax=cbar_ax,
                            format=mpl.ticker.FuncFormatter(self._fmt_solarmass))
        cbar.set_label(r"Dynamical Ejecta Mass ($M_{ej}$)", size=15, labelpad=10)
        fig.text(0.5, 0, "Time since merger (Days)", ha='center')
        fig.text(-0.01, 0.5, r"Log Flux $Erg s^{-1} cm^{-2}A^{-1}$", va='center', rotation='vertical')
        print(
            f"[STATUS] Plotting for mejdyn: {self.mejdyn_range}, mejwind: {self.mejwind_range}, phi: {self.phi_range}, viewing_angle: {self.iobs_range}")

    def select_mejwind(self, wv_range, iobs_range, phi_range, mejdyn_range):
        """Slice reference data set by wind ejecta mass (mejwind).
        >>> data = LightCurve("reference.csv")
        >>> wv_range = [900]
        >>> iobs_range = [0]
        >>> mejdyn_range = [0.01]
        >>> phi_range = [45]
        >>> data.select_mejwind(wv_range, iobs_range, phi_range, mejdyn_range)
        >>> data.mejwind.iloc[0,3]
        2.465e-06
        >>> data.mejwind.iloc[44,2]
        6.077e-05
        >>> data.mejwind.iloc[65,6]
        7.3339e-05
        >>> data.mejwind.shape
        (100, 7)
        >>> data.phi_range
        [45]
        >>> data.mejwind_range
        [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]
        >>> data.iobs_range
        [0]
        >>> data.mejdyn_range
        [0.01]
        """
        mejwind_range_list = self.reference.mejwind.unique()
        self.mejwind_range = sorted(mejwind_range_list)
        self.wv_range = wv_range
        self.iobs_range = iobs_range
        self.phi_range = phi_range
        self.mejdyn_range = mejdyn_range
        mejwind_range_list = [[x] for x in mejwind_range_list]
        df = pd.DataFrame()
        self.select_curve(phiRange=phi_range, mejdynRange=mejdyn_range)
        tempReference = self.selected

        for i in range(len(mejwind_range_list)):
            self.select_curve(mejwindRange=[self.mejwind_range[i]])
            self.extract_curve()
            arr = self.curve[self._compute_wavelength(wv_range[0])][
                iobs_range[0]]  # at wv 900, iobs0, over TIME STEP, for mejdyn 0.01
            assert len(arr) == self.Ntime[0]
            df[self.mejwind_range[i]] = arr
            self.selected = tempReference

        df.columns.name = "mejwind"
        df.index.name = "time_step"
        self.mejwind = df
        self.mejwind = self.mejwind.sort_index(axis=1)

    def plot_mejwind(self, verbose=False):
        """ Plots the data acquired in the wind ejecta mass.
        >>> data = LightCurve("reference.csv")
        >>> wv_range = [900]
        >>> iobs_range = [0]
        >>> mejdyn_range = [0.01]
        >>> phi_range = [45]
        >>> data.select_mejwind(wv_range, iobs_range, phi_range, mejdyn_range)
        >>> data.plot_mejwind() #doctest: +SKIP
        """
        numRows = 2
        numCols = 4
        fig, axes = plt.subplots(nrows=numRows, ncols=numCols, dpi=300, figsize=(6, 3.5))
        plt.tight_layout()
        viewing_angles = np.linspace(0, 1, self.Nobs, endpoint=True)
        colors = plt.cm.BrBG(np.linspace(0, 1, len(self.mejwind_range)))
        counter = 0
        namesList = list(map(str, self.mejwind_range))
        ticks = np.arange(min(self.time_arr), max(self.time_arr) + 1, 5)
        wvList = [self.uBand, self.gBand, self.rBand, self.iBand,
                  self.zBand, self.yBand, self.jBand, self.hBand]
        namesList = ["uBand", "gBand", "rBand", "iBand", "zBand", "yBand", "jBand", "hBand"]

        row = 0
        col = 0

        for row in tqdm(range(0, numRows), disable=not verbose):
            for col in range(0, numCols):
                wv = wvList[counter]
                self.select_mejwind([wv], self.iobs_range, self.phi_range, self.mejdyn_range)
                self.wavelength = self._compute_wavelength(wv)

                for i in range(len(self.mejwind_range)):
                    if i == 0:

                        labelStr = f"{namesList[counter][0].upper()}"
                    else:
                        labelStr = f""
                    axes[row, col].plot(self.time_arr, self.mejwind[self.mejwind_range[i]], label=labelStr,
                                        linewidth=1, color=colors[i])
                axes[row, col].set_yscale('log')
                axes[row, col].legend(handletextpad=-2.0, handlelength=0)
                axes[row, col].axes.get_yaxis().set_visible(False)
                axes[row, col].set_xticks(ticks)

                counter += 1

        self.counter_final = counter
        bottom, top = 0.1, 0.9
        left, right = 0.2, 0.8

        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        my_cmap = "BrBG"
        sm = plt.cm.ScalarMappable(cmap=my_cmap,
                                   norm=plt.Normalize(vmin=min(self.mejwind_range), vmax=max(self.mejwind_range)))
        cbar_ax = fig.add_axes([1, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), extend='both', cax=cbar_ax,
                            format=mpl.ticker.FuncFormatter(self._fmt_solarmass))
        cbar.set_label(r"Wind Ejecta Mass ($M_{ej}$)", size=15, labelpad=10)
        fig.text(0.5, 0, "Time since merger (Days)", ha='center')
        fig.text(-0.01, 0.5, r"Log Flux $Erg s^{-1} cm^{-2}A^{-1}$", va='center', rotation='vertical')
        print(
            f"[STATUS] Plotting for mejdyn: {self.mejdyn_range}, mejwind: {self.mejwind_range}, phi: {self.phi_range}, viewing_angle: {self.iobs_range}")

    def select_phi(self, wv_range, iobs_range, mejwind_range, mejdyn_range):
        """Slice reference data set by half-seperation angle (phi).
        >>> data = LightCurve("reference.csv")
        >>> wv_range = [900]
        >>> iobs_range = [0]
        >>> mejdyn_range = [0.01]
        >>> mejwind_range = [0.13]
        >>> data.select_phi(wv_range, iobs_range, mejwind_range, mejdyn_range)
        >>> data.phi.iloc[0,3]
        1.203e-06
        >>> data.phi.iloc[44,2]
        0.00023286
        >>> data.phi.iloc[65,4]
        6.9335e-05
        >>> data.phi.shape
        (100, 7)
        >>> data.phi_range
        [0, 15, 30, 45, 60, 75, 90]
        >>> data.mejwind_range
        [0.13]
        >>> data.iobs_range
        [0]
        >>> data.mejdyn_range
        [0.01]
        """
        phi_range_list = self.reference.phi.unique()
        self.phi_range = sorted(phi_range_list)
        self.wv_range = wv_range
        self.iobs_range = iobs_range
        self.mejwind_range = mejwind_range
        self.mejdyn_range = mejdyn_range

        phi_range_list = [[x] for x in phi_range_list]
        df = pd.DataFrame()
        self.select_curve(mejwindRange=mejwind_range, mejdynRange=mejdyn_range)
        tempReference = self.selected

        for i in range(len(phi_range_list)):
            self.select_curve(phiRange=[self.phi_range[i]])
            self.extract_curve()
            arr = self.curve[self._compute_wavelength(wv_range[0])][
                iobs_range[0]]  # at wv 900, iobs0, over TIME STEP, for mejdyn 0.01
            assert len(arr) == self.Ntime[0]
            df[self.phi_range[i]] = arr
            self.selected = tempReference

        df.columns.name = "phi"
        df.index.name = "time_step"
        self.phi = df
        self.phi = self.phi.sort_index(axis=1)

    def plot_phi(self, verbose=False):
        """ Plots the data acquired in the half-opening angle.
        >>> dat = LightCurve("reference.csv")
        >>> wv_range = [900]
        >>> iobs_range = [0]
        >>> mejdyn_range = [0.01]
        >>> mejwind_range = [0.13]
        >>> dat.select_phi(wv_range, iobs_range, mejwind_range, mejdyn_range)
        >>> dat.plot_phi() #doctest: +SKIP
        """
        numRows = 2
        numCols = 4
        fig, axes = plt.subplots(nrows=numRows, ncols=numCols, dpi=300, figsize=(6, 3.5))
        plt.tight_layout()
        viewing_angles = np.linspace(0, 1, self.Nobs, endpoint=True)
        colors = plt.cm.PuOr(np.linspace(0, 1, len(self.phi_range)))
        counter = 0
        namesList = list(map(str, self.phi_range))
        ticks = np.arange(min(self.time_arr), max(self.time_arr) + 1, 5)
        wvList = [self.uBand, self.gBand, self.rBand, self.iBand,
                  self.zBand, self.yBand, self.jBand, self.hBand]
        namesList = ["uBand", "gBand", "rBand", "iBand", "zBand", "yBand", "jBand", "hBand"]
        for row in tqdm(range(0, numRows), disable=not verbose):
            for col in range(0, numCols):
                wv = wvList[counter]
                self.select_phi([wv], self.iobs_range, self.mejwind_range, self.mejdyn_range)
                self.wavelength = self._compute_wavelength(wv)

                for i in range(len(self.phi_range)):
                    if i == 0:

                        labelStr = f"{namesList[counter][0].upper()}"
                    else:
                        labelStr = f""
                    axes[row, col].plot(self.time_arr, self.phi[self.phi_range[i]], label=labelStr,
                                        linewidth=1, color=colors[i])
                axes[row, col].set_yscale('log')
                axes[row, col].legend(handletextpad=-2.0, handlelength=0)
                axes[row, col].axes.get_yaxis().set_visible(False)
                axes[row, col].set_xticks(ticks)

                counter += 1
        self.counter_final = counter
        bottom, top = 0.1, 0.9
        left, right = 0.2, 0.8

        fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)
        my_cmap = "PuOr"
        sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=min(self.phi_range), vmax=max(self.phi_range)))
        cbar_ax = fig.add_axes([1, bottom, 0.03, top - bottom])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), extend='both', cax=cbar_ax,
                            format=mpl.ticker.FuncFormatter(self._fmt_degree))
        cbar.set_label(r"Phi ($\Phi$)", size=15, labelpad=10)
        fig.text(0.5, 0, "Time since merger (Days)", ha='center')
        fig.text(-0.01, 0.5, r"Log Flux $Erg s^{-1} cm^{-2}A^{-1}$", va='center', rotation='vertical')
        print(
            f"[STATUS] Plotting for mejdyn: {self.mejdyn_range}, mejwind: {self.mejwind_range}, phi: {self.phi_range}, viewing_angle: {self.iobs_range}")

    def _fmt_degree(self, x, pos):
        """Formats the text given into scientific notation. Used as a helper function in plotting.
        """

        return r'${}^o$'.format(round(x))

    def _fmt_solarmass(self, x, pos):
        """Formats the text given into scientific notation. Used as a helper function in plotting.
        """
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}} M_\odot$'.format(a, b)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
