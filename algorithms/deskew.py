import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


class Deskew:

    def __init__(self, time, vd_values, is_values, on_1, on_2, off_1, auto_deskew, manual_deskew, deskew_time, lshunt, Is_th):
        self.time = time
        self.vd = vd_values
        self.is_ = is_values
        self.on_1_time = on_1['time']
        self.on_2_time = on_2['time']
        self.off_1_time = off_1['time']
        self.on_1_idx = on_1['thresh_idx']
        self.on_2_idx = on_2['thresh_idx']
        self.off_1_idx = off_1['thresh_idx']
        self.lshunt = lshunt
        self.Is_th = Is_th

        if auto_deskew:
            self.deskew()
            self.is_updated = np.interp(self.time + self.deskew_time, self.time, self.is_)
            self.off_1_time_updated = self.off_1_time - self.deskew_time
            self.off_1_idx_updated = np.argmin(np.abs(self.time - self.off_1_time_updated))
            self.off_2_idx = self.idx_Is_0()
            self.off_2_time = self.time[self.off_2_idx]
        elif manual_deskew:
            self.deskew()
            if abs(deskew_time) > 10e-9:
                deskew_time = 10e-9
            self.deskew_time = deskew_time
            self.is_updated = np.interp(self.time + self.deskew_time, self.time, self.is_)
            self.off_1_time_updated = self.off_1_time - self.deskew_time
            self.off_1_idx_updated = np.argmin(np.abs(self.time - self.off_1_time_updated))
            self.off_2_idx = self.idx_Is_0()
            self.off_2_time = self.time[self.off_2_idx]
        else:
            self.deskew()
            self.deskew_time = 0
            self.is_updated = self.is_
            self.off_1_time_updated = self.off_1_time
            self.off_1_idx_updated = self.off_1_idx
            self.off_2_idx = self.idx_Is_0()
            self.off_2_time = self.time[self.off_2_idx]

    def deskew(self):
        vd_smoothed = gaussian_filter1d(self.vd, sigma=1)
        is_derivative = np.gradient(self.is_, self.time)
        lenth = round((self.on_2_idx - self.on_1_idx)/2)
        idx_start = self.on_1_idx
        idx_end = idx_start + lenth + 1
        try:
            idx_current_peak = np.argmax(is_derivative[idx_start - 10:idx_end]) + idx_start - 10

            vd_stable = np.average(vd_smoothed[idx_start - 20:idx_start - 10])
            is_derivative_stable = np.average(is_derivative[idx_start - 20:idx_start - 10])

            delta_v = vd_stable - vd_smoothed[idx_current_peak]
            delta_i = is_derivative_stable - is_derivative[idx_current_peak]
            if self.lshunt is None:
                self.lshunt = delta_v/delta_i
            self.constant = vd_stable - self.lshunt*is_derivative_stable

            popt, pcov = curve_fit(self.func_fitting, self.time[idx_start - 10:idx_current_peak], self.vd[idx_start - 10:idx_current_peak], p0=[-2e-9])
            self.deskew_time = popt[0]
            perr = np.sqrt(np.diag(pcov))
            self.std = perr[0]
        except:
            self.lshunt = 0
            self.constant = 0
            self.deskew_time = 0
            self.std = 0

    def func_fitting(self, t, t0):
        return self.lshunt*self.is_derivative(t + t0) + self.constant

    def is_derivative(self, t):
        dis_dt = np.gradient(self.is_, self.time)
        return np.interp(t, self.time, dis_dt)

    def idx_Is_0(self):
        idx = self.off_1_idx
        Is = self.is_updated[idx]
        while Is <= 0:
            idx += 1
            Is = self.is_updated[idx]

        while Is > self.Is_th:
            idx += 1
            Is = self.is_updated[idx]

        return idx
