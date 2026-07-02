"""
This is the main py file to process tss files. It will read the tss files and output switching timing (tdon, tdoff, tr, tf) and energy (Eon and Eoff). It can also generate figures for IV, Vd-Vg, switching power and switching energy.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import configparser
from pathlib import Path

from .tss_file import TSSFile                # Module to read tss files
from .wfm_edge import WFM_edge               # Module to get waveform edge information (such as turn-on/off time)
from .deskew import Deskew                   # Module to do time deskewing if needed, and to get turn-off time when Is = 0 A
from .tf_correction import get_fall_time


class TSS_process:
    """Read the tss file."""
    def __init__(self, folder, tss_file, config_file):

        tss = TSSFile(folder, tss_file, source='Local')
        self.channel_labels = tss.channel_labels

        self._set_channels()
        config = configparser.ConfigParser()

        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parents[1] / 'config_files' / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config.read(config_path)
        self.time_params = {key: config.get("Time_parameters", key) for key in config["Time_parameters"]}
        self.energy_params = {key: config.get("Energy_parameters", key) for key in config["Energy_parameters"]}
        self.qg_params = {key: config.get("Qg_parameters", key) for key in config["Qg_parameters"]}

        self.time = tss.waveforms[self.vd_channel].time_for_frame()

        def get_values(channel):
            return tss.waveforms[channel].values_for_frame(0) if channel is not None else None

        self.vd_values = get_values(self.vd_channel)
        self.vg_values = get_values(self.vg_channel)
        self.is_values = get_values(self.is_channel)
        self.il_values = get_values(self.il_channel)
        self.vdc_values = get_values(self.vdc_channel)
        self.vcl_values = get_values(self.vcl_channel)
        self.vgi_values = get_values(self.vgi_channel)
        self.ig_values = get_values(self.ig_channel)
        self.mc_gate_values = get_values(self.mc_gate_channel)

    def time_extraction(self, noise_flag=False):
        '''Calculate switching timing (tdon, tdoff, tr, tf).'''
        if self.vd_channel is None or self.vg_channel is None:
            raise Exception("Cannot find Vd or Vg channel!")

        vd_edges = WFM_edge(self.vd_values, self.time, falling_edge_number=eval(self.time_params['falling_edge_number']), 
                            rising_edge_number=eval(self.time_params['rising_edge_number']), 
                            peak_width=eval(self.time_params['peak_width']),
                            peak_distance=eval(self.time_params['peak_distance']), 
                            sigma=eval(self.time_params['sigma']),
                            margin_ratio=eval(self.time_params['margin_ratio']),)

        vg_edges = WFM_edge(self.vg_values, self.time, falling_edge_number=eval(self.time_params['falling_edge_number']), 
                            rising_edge_number=eval(self.time_params['rising_edge_number']), 
                            peak_width=eval(self.time_params['peak_width']),
                            peak_distance=eval(self.time_params['peak_distance']), 
                            sigma=eval(self.time_params['sigma']),
                            margin_ratio=eval(self.time_params['margin_ratio']),)

        time_thresh_1 = eval(self.time_params['time_thresh_1'])
        time_thresh_2 = eval(self.time_params['time_thresh_2'])
        self.vd_f1 = vd_edges.get_edge(2, 'falling', time_thresh_2)['time']
        edge = vd_edges.get_edge(2, 'falling', time_thresh_1)
        vd_f2_results = get_fall_time(self.vd_values, self.time, 25, edge, vd_edges)
        self.vd_f2 = vd_f2_results['time']

        self.vd_r1 = vd_edges.get_edge(1, 'rising', time_thresh_1)['time']
        self.vd_r2 = vd_edges.get_edge(1, 'rising', time_thresh_2)['time']
        self.vg_r1 = vg_edges.get_edge(2, 'rising', time_thresh_1, tdon_correction=self.time_params['tdon_correction'],
                                        switching_time_factor=eval(self.time_params['switching_time_factor']))['time']
        self.vg_f1 = vg_edges.get_edge(1, 'falling', time_thresh_2)['time']

        tdon = (self.vd_f1 - self.vg_r1)*1e9
        tf = (self.vd_f2 - self.vd_f1)*1e9
        tdoff = (self.vd_r1 - self.vg_f1)*1e9
        tr = (self.vd_r2 - self.vd_r1)*1e9

        return [tdon, tf, tdoff, tr]

    def energy_extraction(self, auto_deskew=False, manual_deskew=False, deskew_time=0, lshunt=None, Is_th=0):
        '''Calculate switching energy (Eon and Eoff).'''
        if self.vd_channel is None or self.is_channel is None:
            raise Exception("Cannot find Vd or Is channel!")

        vd_edges = WFM_edge(self.vd_values, self.time, falling_edge_number=eval(self.energy_params['falling_edge_number']), 
                            rising_edge_number=eval(self.energy_params['rising_edge_number']), 
                            peak_width=eval(self.energy_params['peak_width']),
                            peak_distance=eval(self.energy_params['peak_distance']), 
                            sigma=eval(self.energy_params['sigma']),
                            margin_ratio=eval(self.energy_params['margin_ratio']),)

        eon_thresh_1 = eval(self.energy_params['eon_thresh_1'])
        eon_thresh_2 = eval(self.energy_params['eon_thresh_2'])
        eoff_thresh_1 = eval(self.energy_params['eoff_thresh_1'])
        Is_th = eval(self.energy_params['eoff_thresh_2'])

        on_1 = vd_edges.get_edge(2, 'falling', eon_thresh_1)
        on_2 = vd_edges.get_edge(2, 'falling', eon_thresh_2)
        off_1 = vd_edges.get_edge(1, 'rising', eoff_thresh_1)

        self.on_1_idx = on_1['thresh_idx']
        self.on_2_idx = on_2['thresh_idx']
        self.off_1_idx = off_1['thresh_idx']
        self.on_1_time = on_1['time']
        self.on_2_time = on_2['time']
        self.off_1_time = off_1['time']

        on_1_modified = vd_edges.get_edge(2, 'falling', 95)
        dyp = Deskew(self.time, self.vd_values, self.is_values, on_1_modified, on_2, off_1, auto_deskew, manual_deskew, deskew_time, lshunt, Is_th)
        self.deskew_time = dyp.deskew_time
        self.lshunt = dyp.lshunt
        self.constant = dyp.constant
        self.std_deskew = dyp.std

        self.off_2_idx = dyp.off_2_idx
        self.off_2_time = dyp.off_2_time
        self.is_updated = dyp.is_updated

        self.power_values = np.multiply(self.vd_values, self.is_updated)

        eon = np.trapz(self.power_values[self.on_1_idx:self.on_2_idx], self.time[self.on_1_idx:self.on_2_idx])
        eoff = np.trapz(self.power_values[self.off_1_idx:self.off_2_idx], self.time[self.off_1_idx:self.off_2_idx])

        return [eon*1e6, eoff*1e6, self.deskew_time*1e9, self.lshunt*1e9]

    def gate_charge_extraction(self):
        if self.vg_channel is None or self.ig_channel is None:
            raise Exception("Cannot find Vg or Ig channel!")

        vg_edges = WFM_edge(self.vg_values, self.time, falling_edge_number=eval(self.qg_params['falling_edge_number']), 
                            rising_edge_number=eval(self.qg_params['rising_edge_number']), 
                            peak_width=eval(self.qg_params['peak_width']),
                            peak_distance=eval(self.qg_params['peak_distance']), 
                            sigma=eval(self.qg_params['sigma']),
                            margin_ratio=eval(self.qg_params['margin_ratio']),)

        qgoff_thresh_1 = eval(self.qg_params['qgoff_thresh_1'])
        qgoff_thresh_2 = eval(self.qg_params['qgoff_thresh_2'])
        qgon_thresh_1 = eval(self.qg_params['qgon_thresh_1'])
        qgon_thresh_2 = eval(self.qg_params['qgon_thresh_2'])

        idx_t1 = vg_edges.get_edge(1, 'falling', qgoff_thresh_1)['thresh_idx']
        idx_t2 = vg_edges.get_edge(1, 'falling', qgoff_thresh_2)['thresh_idx']
        idx_t3 = vg_edges.get_edge(2, 'rising', qgon_thresh_1)['thresh_idx']
        idx_t4 = vg_edges.get_edge(2, 'rising', qgon_thresh_2)['thresh_idx']

        self.t1 = self.time[idx_t1]
        self.t2 = self.time[idx_t2]
        self.t3 = self.time[idx_t3]
        self.t4 = self.time[idx_t4]

        qgoff = np.trapz(self.ig_values[idx_t1:idx_t2], self.time[idx_t1:idx_t2])
        qgon = np.trapz(self.ig_values[idx_t3:idx_t4], self.time[idx_t3:idx_t4])

        return [qgoff*1e9, qgon*1e9]

    def dvdt_extraction(self, plot_figures=False):
        vd_edges = WFM_edge(self.vd_values, self.time, falling_edge_number=eval(self.time_params['falling_edge_number']), 
                            rising_edge_number=eval(self.time_params['rising_edge_number']), 
                            peak_width=eval(self.time_params['peak_width']),
                            peak_distance=eval(self.time_params['peak_distance']), 
                            sigma=eval(self.time_params['sigma']),
                            margin_ratio=eval(self.time_params['margin_ratio']),)

        idx_on_1 = vd_edges.get_edge(2, 'falling', 90)['thresh_idx']
        idx_on_2 = vd_edges.get_edge(2, 'falling', 10)['thresh_idx']
        idx_off_1 = vd_edges.get_edge(1, 'rising', 10)['thresh_idx']
        idx_off_2 = vd_edges.get_edge(1, 'rising', 90)['thresh_idx']

        time_turn_on = self.time[idx_on_1:idx_on_2]
        vd_turn_on = self.vd_values[idx_on_1:idx_on_2]
        time_turn_off = self.time[idx_off_1:idx_off_2]
        vd_turn_off = self.vd_values[idx_off_1:idx_off_2]

        dvdt_on = np.gradient(vd_turn_on, time_turn_on)
        dvdt_on_average = (self.vd_values[idx_on_1] - self.vd_values[idx_on_2]) / (self.time[idx_on_1] - self.time[idx_on_2])
        dvdt_on_peak = min(dvdt_on)

        dvdt_off = np.gradient(vd_turn_off, time_turn_off)
        dvdt_off_average = (self.vd_values[idx_off_1] - self.vd_values[idx_off_2]) / (self.time[idx_off_1] - self.time[idx_off_2])
        dvdt_off_peak = max(dvdt_off)

        if plot_figures:
            self._dvdt_figures(idx_on_1, idx_on_2, idx_off_1, idx_off_2)

        return [dvdt_on_average*1e-9, dvdt_on_peak*1e-9, dvdt_off_average*1e-9, dvdt_off_peak*1e-9]

    def _dvdt_figures(self, idx_on_1, idx_on_2, idx_off_1, idx_off_2):
        xmin = self.time[idx_on_1]*1e9 - 50
        xmax = self.time[idx_on_2]*1e9 + 50
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin*1e-9, xmax*1e-9)
        plt.plot(self.time*1e9, self.vd_values)
        plt.vlines(self.time[idx_on_1]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.vlines(self.time[idx_on_2]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        plt.xlabel('Time (ns)')
        plt.ylabel('Vd (V)')
        plt.grid()
        plt.title('Vd turn on window')
        plt.show()

        xmin = self.time[idx_off_1]*1e9 - 50
        xmax = self.time[idx_off_2]*1e9 + 50
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin*1e-9, xmax*1e-9)
        plt.plot(self.time*1e9, self.vd_values)
        plt.vlines(self.time[idx_off_1]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.vlines(self.time[idx_off_2]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        plt.xlabel('Time (ns)')
        plt.ylabel('Vd (V)')
        plt.grid()
        plt.title('Vd turn off window')
        plt.show()

    def rdson_extraction(self, t_trigger, t_avg, t_interpulse, t_dead, t_2nd):
        pre_window_1 = t_trigger - t_avg - 100e-9
        pre_window_2 = t_trigger - 100e-9
        pos_window_1 = t_trigger + t_interpulse + t_dead + t_2nd - t_avg - 100e-9
        pos_window_2 = t_trigger + t_interpulse + t_dead + t_2nd - 100e-9
        windows = [pre_window_1, pre_window_2, pos_window_1, pos_window_2]

        results = []
        pre_results = []
        pos_results = []
        idx = []
        for time_window in windows:
            idx.append(np.argmin(np.abs(self.time - time_window)))

        pre_results.append(np.average(self.is_values[idx[0]:idx[1]]))
        pos_results.append(np.average(self.is_values[idx[2]:idx[3]]))
        pre_results.append(np.average(self.vcl_values[idx[0]:idx[1]]))
        pos_results.append(np.average(self.vcl_values[idx[2]:idx[3]]))

        results = [
            pre_results[1]/pre_results[0] * 1000,
            pos_results[1]/pos_results[0] * 1000,
        ]
        return results

    def P42016_extraction(self, plot_figures=False):
        vd_edges = WFM_edge(self.vd_values, self.time)
        idx_on_1 = vd_edges.get_edge(2, 'falling', 99)['thresh_idx']
        idx_on_2 = vd_edges.get_edge(2, 'falling', 1)['thresh_idx']
        idx_off_1 = vd_edges.get_edge(1, 'rising', 1)['thresh_idx']
        idx_off_2 = vd_edges.get_edge(1, 'rising', 99)['thresh_idx']

        window_on_1 = int(idx_on_1 - 1*(idx_on_2 - idx_on_1))
        window_on_2 = int(idx_on_2 + 1*(idx_on_2 - idx_on_1))
        window_off_1 = int(idx_off_1 - 1*(idx_off_2 - idx_off_1))
        window_off_2 = int(idx_off_2 + 5*(idx_off_2 - idx_off_1))

        is_on_edges = WFM_edge(self.is_values[window_on_1:window_on_2], self.time[window_on_1:window_on_2])
        is_off_edges = WFM_edge(self.is_values[window_off_1:window_off_2], self.time[window_off_1:window_off_2])

        idx_is_on_1 = is_on_edges.get_edge(1, 'rising', 10)['thresh_idx'] + window_on_1
        idx_is_on_2 = is_on_edges.get_edge(1, 'rising', 90)['thresh_idx'] + window_on_1
        idx_is_off_1 = is_off_edges.get_edge(1, 'falling', 90)['thresh_idx'] + window_off_1
        idx_is_off_2 = is_off_edges.get_edge(1, 'falling', 10)['thresh_idx'] + window_off_1

        didt_on_average = (self.is_values[idx_is_on_1] - self.is_values[idx_is_on_2]) / (self.time[idx_is_on_1] - self.time[idx_is_on_2])
        didt_off_average = (self.is_values[idx_is_off_1] - self.is_values[idx_is_off_2]) / (self.time[idx_is_off_1] - self.time[idx_is_off_2])

        # time_mcgate = np.interp(1.5, self.mc_gate_values[idx_is_off_1:idx_is_off_2], self.time[idx_is_off_1:idx_is_off_2])
        # is_at_mcgate = np.interp(time_mcgate, self.time[idx_is_off_1:idx_is_off_2], self.is_values[idx_is_off_1:idx_is_off_2])

        # if plot_figures:
            # self._P42016_figures(idx_is_on_1, idx_is_on_2, idx_is_off_1, idx_is_off_2, time_mcgate)

        # return [didt_on_average*1e-9, didt_off_average*1e-9, is_at_mcgate]
        return [didt_on_average*1e-9, didt_off_average*1e-9]

    def _P42016_figures(self, idx_is_on_1, idx_is_on_2, idx_is_off_1, idx_is_off_2, time_mcgate):
        xmin = self.time[idx_is_on_1]*1e9 - 50
        xmax = self.time[idx_is_on_2]*1e9 + 50
        ymin, ymax = self._autoscale_y(self.time, self.is_values, xmin*1e-9, xmax*1e-9)
        fig, ax1 = plt.subplots()
        ax1.plot(self.time*1e9, self.is_values, 'b')
        ax1.vlines(self.time[idx_is_on_1]*1e9, ymin, ymax, color='green', linestyles='--')
        ax1.vlines(self.time[idx_is_on_2]*1e9, ymin, ymax, color='green', linestyles='--')
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Is (A)', color='b')
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time*1e9, self.mc_gate_values, 'r-')
        ax2.set_ylabel('MC_gate (V)', color='r')
        plt.title('Is turn on window')
        plt.show()

        xmin = self.time[idx_is_off_1]*1e9 - 50
        xmax = self.time[idx_is_off_2]*1e9 + 50
        ymin, ymax = self._autoscale_y(self.time, self.is_values, xmin, xmax)
        fig, ax1 = plt.subplots()
        ax1.plot(self.time*1e9, self.is_values, 'b')
        ax1.vlines(self.time[idx_is_off_1]*1e9, ymin, ymax, color='green', linestyles='--')
        ax1.vlines(self.time[idx_is_off_2]*1e9, ymin, ymax, color='green', linestyles='--')
        # ax1.vlines(time_mcgate*1e9, ymin, ymax, color='red', linestyles='--')
        ax1.set_ylim([ymin, ymax])
        ax1.set_xlim([xmin, xmax])
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Is (A)', color='b')
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time*1e9, self.mc_gate_values, 'r-')
        ax2.set_ylabel('MC_gate (V)', color='r')
        plt.title('Is turn off window')
        plt.show()

    def P42016_vgi_extraction(self, plot_figures=False):
        from scipy.ndimage import gaussian_filter1d
        vgi_smooth = gaussian_filter1d(self.vgi_values, sigma=15)
        vgi_edges = WFM_edge(vgi_smooth, self.time, sigma=5)
        idx_off_1 = vgi_edges.get_edge(1, 'falling', 90)['thresh_idx']
        idx_off_2 = vgi_edges.get_edge(1, 'falling', 10)['thresh_idx']
        idx_on_1 = vgi_edges.get_edge(2, 'rising', 10)['thresh_idx']
        idx_on_2 = vgi_edges.get_edge(2, 'rising', 90)['thresh_idx']

        dvgidt_on_average = (vgi_smooth[idx_on_1] - vgi_smooth[idx_on_2]) / (self.time[idx_on_1] - self.time[idx_on_2])
        dvgidt_off_average = (vgi_smooth[idx_off_1] - vgi_smooth[idx_off_2]) / (self.time[idx_off_1] - self.time[idx_off_2])

        if plot_figures:
            self._P42016_vgi_figures(idx_on_1, idx_on_2, idx_off_1, idx_off_2, vgi_smooth)

        return [dvgidt_on_average*1e-9, dvgidt_off_average*1e-9]

    def _P42016_vgi_figures(self, idx_on_1, idx_on_2, idx_off_1, idx_off_2, vgi_smooth):
        xmin = self.time[idx_on_1]*1e9 - 50
        xmax = self.time[idx_on_2]*1e9 + 50
        ymin, ymax = self._autoscale_y(self.time, self.vgi_values, xmin*1e-9, xmax*1e-9)
        plt.plot(self.time*1e9, self.vgi_values, 'b')
        plt.plot(self.time*1e9, vgi_smooth, 'r', linestyle='--')
        plt.vlines(self.time[idx_on_1]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.vlines(self.time[idx_on_2]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        plt.xlabel('Time (ns)')
        plt.ylabel('Vgi (V)')
        plt.title('Turn-on Vgi waveform')
        plt.grid()
        plt.show()

        xmin = self.time[idx_off_1]*1e9 - 50
        xmax = self.time[idx_off_2]*1e9 + 50
        ymin, ymax = self._autoscale_y(self.time, self.vgi_values, xmin*1e-9, xmax*1e-9)
        plt.plot(self.time*1e9, self.vgi_values, 'b')
        plt.plot(self.time*1e9, vgi_smooth, 'r', linestyle='--')
        plt.vlines(self.time[idx_off_1]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.vlines(self.time[idx_off_2]*1e9, ymin, ymax, color='green', linestyles='--')
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        plt.xlabel('Time (ns)')
        plt.ylabel('Vgi (V)')
        plt.title('Turn-off Vgi waveform')
        plt.grid()
        plt.show()

    def _set_channels(self):
        signal_groups = {
            "vd": ['Vd', 'Vds', 'Vmid'],
            "vg": ['Vg', 'Vgs', 'VgLS', 'Vg1_LS', 'Vg_LS', 'VGO_S1'],
            "is": ['Is', 'Ids', 'Id', 'IsR'],
            "il": ['IL'],
            "vdc": ['DCLink'],
            "vcl": ['Vcl'],
            "ig": ['Ig'],
            'mc_gate': ['MC_GATE'],
            "vgi": ['Vgi'],
        }

        self.vd_channel = self.vg_channel = self.is_channel = self.vgi_channel = None
        self.il_channel = self.vdc_channel = self.vcl_channel = self.ig_channel = self.mc_gate_channel = None

        for key, value in self.channel_labels.items():
            for attr, labels in signal_groups.items():
                if value in labels:
                    setattr(self, f"{attr}_channel", key)
                    break

    def _get_idx(self, xmin, xmax):
        '''Get the closet index for xmin and xmax in time array.'''
        xmin_idx = np.argmin(np.abs(self.time - xmin))
        xmax_idx = np.argmin(np.abs(self.time - xmax))
        return xmin_idx, xmax_idx

    def _autoscale_y(self, x, y, xmin, xmax, margin = 0.1):
        """Auto scale y axis"""
        min_idx = np.argmin(np.abs(x - xmin))
        max_idx = np.argmin(np.abs(x - xmax))

        y_top = max(y[min_idx:max_idx])
        y_bot = min(y[min_idx:max_idx])

        l = y_top - y_bot
        ymax = y_top + l*margin
        ymin = y_bot - l*margin

        return ymin, ymax

    def _func_time(self, y, pos):
        time = y*1e9
        return "{:.0f}".format((time))

    def _func_energy(self, y, pos):
        energy = y*1e6
        return "{:.0f}".format((energy))

    def figure_on_off_VV(self):
        '''Plot Vd-Vg waveforms for turn-on and turn-off.'''
        xmin = min(self.vg_r1, self.vd_f2) - 20e-9
        xmax = max(self.vg_r1, self.vd_f2) + 50e-9
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin, xmax)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vd_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vd (V)', color='b')
        ax1.vlines(self.vg_r1, ymin, ymax, color='green')
        ax1.vlines(self.vd_f1, ymin, ymax, color='green')
        ax1.vlines(self.vd_f2, ymin, ymax, color='green')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.vg_values, 'r-')
        ax2.set_ylabel('Vg (V)', color='r')
        plt.xlim([xmin, xmax])
        plt.title('Turn-on waveform')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.show()

        xmin = self.vg_f1 - 30e-9
        xmax = self.vd_r2 + 30e-9
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin, xmax)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vd_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vd (V)', color='b')
        ax1.vlines(self.vg_f1, ymin, ymax, color='green')
        ax1.vlines(self.vd_r1, ymin, ymax, color='green')
        ax1.vlines(self.vd_r2, ymin, ymax, color='green')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.vg_values, 'r-')
        ax2.set_ylabel('Vg (V)', color='r')
        plt.xlim([xmin, xmax])
        plt.title('Turn-off waveform')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.show()

    def figure_on_IV(self, figure_show=True, pdf_save=False, pdf=None):
        '''Plot IV waveforms for turn-on and turn-off.'''
        xmin = self.on_1_time - 10e-9
        xmax = self.on_2_time + 20e-9
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin, xmax)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vd_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vd (V)', color='b')
        ax1.vlines(self.on_1_time, ymin, ymax, color='green')
        ax1.vlines(self.on_2_time, ymin, ymax, color='green')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.is_updated, 'r-')
        ax2.set_ylabel('Is (A)', color='r')
        plt.xlim([xmin, xmax])
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.title('Turn-on waveform')
        if figure_show:
            plt.show()
        if pdf_save:
            pdf.savefig()
            plt.close()

    def figure_off_IV(self, figure_show=True, pdf_save=False, pdf=None):
        xmin = self.off_1_time - 10e-9
        xmax = self.off_2_time + 20e-9
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin, xmax)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vd_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vd (V)', color='b')
        ax1.vlines(self.off_1_time, ymin, ymax, color='green')
        ax1.vlines(self.off_2_time, ymin, ymax, color='green')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.is_updated, 'r-')
        ax2.set_ylabel('Is (A)', color='r')
        plt.xlim([xmin, xmax])
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.title('Turn-off waveform')
        if figure_show:
            plt.show()
        if pdf_save:
            pdf.savefig()
            plt.close()

    def figure_before_deskew(self, figure_show=True, pdf_save=False, pdf=None):
        '''Plot derivative of Is and Vd during turn-on before time deskewing.'''
        xmin = self.on_1_time - 10e-9
        xmax = self.on_2_time
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin, xmax, margin=0.1)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vd_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vd (V)', color='b')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()
        ax2 = ax1.twinx()
        ax2.plot(self.time, self.lshunt*np.gradient(self.is_values, self.time) + self.constant, 'r-')
        ax2.set_ylabel('-L*dIs/dt + C', color='r')
        ax2.set_ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.figtext(0.15, 0.2, f'Lloop = {round(self.lshunt*-1e9, 2)} (nH)', fontsize=10)
        plt.figtext(0.15, 0.15, f'Deskew time = 0 (ns)', fontsize=10)
        plt.title('Figure before manual deskewing (Turn-on)')
        if figure_show:
            plt.show()
        if pdf_save:
            pdf.savefig()
            plt.close()

    def figure_time_deskew(self, figure_show=True, pdf_save=False, pdf=None):
        '''Plot derivative of Is and Vd during turn-on after time deskewing.'''
        xmin = self.on_1_time - 10e-9
        xmax = self.on_2_time
        ymin, ymax = self._autoscale_y(self.time, self.vd_values, xmin, xmax, margin=0.1)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vd_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vd (V)', color='b')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()
        ax2 = ax1.twinx()
        ax2.plot(self.time, self.lshunt*np.gradient(self.is_updated, self.time) + self.constant, 'r-')
        ax2.set_ylabel('-L*dIs/dt + C', color='r')
        ax2.set_ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.figtext(0.15, 0.2, f'Lloop = {round(self.lshunt*-1e9, 2)} (nH)', fontsize=10)
        plt.figtext(0.15, 0.15, f'Deskew time = {self.deskew_time*1e9} (ns)', fontsize=10)
        plt.title('Figure after manual deskewing (Turn-on)')
        if figure_show:
            plt.show()
        if pdf_save:
            pdf.savefig()
            plt.close()

    def figure_on_energy(self, figure_show=True, pdf_save=False, pdf=None):
        '''Plot energy flow vs time for turn-on and turn-off.'''
        xmin = self.on_1_time - 10e-9
        xmax = self.on_2_time + 20e-9

        xmin_idx, xmax_idx = self._get_idx(xmin, xmax)
        time = self.time[xmin_idx:xmax_idx]
        power = self.power_values[xmin_idx:xmax_idx]
        energy_values = np.zeros_like(time)
        for i in range(len(time)):
            energy_values[i] = np.trapz(power[:i+1], time[:i+1])

        ymin, ymax = self._autoscale_y(time, energy_values, xmin, xmax, margin=0.15)

        plt.plot(time, energy_values)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.vlines(self.on_1_time, ymin, ymax, color='green')
        plt.vlines(self.on_2_time, ymin, ymax, color='green')
        plt.xlabel('Time (ns)')
        plt.ylabel('Energy (uJ)')
        plt.grid()
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self._func_energy))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.figtext(0.6, 0.12, f'Deskew time = {self.deskew_time*1e9} (ns)', fontsize=10)
        plt.title('Turn-on energy')
        if figure_show:
            plt.show()
        if pdf_save:
            pdf.savefig()
            plt.close()

    def figure_off_energy(self, figure_show=True, pdf_save=False, pdf=None):
        xmin = self.off_1_time - 10e-9
        xmax = self.off_2_time + 20e-9

        xmin_idx, xmax_idx = self._get_idx(xmin, xmax)
        time = self.time[xmin_idx:xmax_idx]
        power = self.power_values[xmin_idx:xmax_idx]
        energy_values = np.zeros_like(time)
        for i in range(len(time)):
            energy_values[i] = np.trapz(power[:i+1], time[:i+1])

        ymin, ymax = self._autoscale_y(time, energy_values, xmin, xmax, margin=0.15)

        plt.plot(time, energy_values)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.vlines(self.off_1_time, ymin, ymax, color='green')
        plt.vlines(self.off_2_time, ymin, ymax, color='green')
        plt.xlabel('Time (ns)')
        plt.ylabel('Energy (uJ)')
        plt.grid()
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self._func_energy))
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.figtext(0.6, 0.12, f'Deskew time = {self.deskew_time*1e9} (ns)', fontsize=10)
        plt.title('Turn-off energy')
        if figure_show:
            plt.show()
        if pdf_save:
            pdf.savefig()
            plt.close()

    def figure_gate_waveform(self):
        xmin = self.t1 - 30e-9
        xmax = self.t2 + 100e-9
        ymin, ymax = self._autoscale_y(self.time, self.vg_values, xmin, xmax)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vg_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vg (V)', color='b')
        ax1.vlines(self.t1, ymin, ymax, color='green')
        ax1.vlines(self.t2, ymin, ymax, color='green')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.ig_values*1e3, 'r-')
        ax2.set_ylabel('Ig (mA)', color='r')
        plt.xlim([xmin, xmax])
        plt.title('Turn-off waveform')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.show()

        xmin = self.t3 - 30e-9
        xmax = self.t4 + 100e-9
        ymin, ymax = self._autoscale_y(self.time, self.vg_values, xmin, xmax)

        fig, ax1 = plt.subplots()
        ax1.plot(self.time, self.vg_values, 'b-')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Vg (V)', color='b')
        ax1.vlines(self.t3, ymin, ymax, color='green')
        ax1.vlines(self.t4, ymin, ymax, color='green')
        ax1.set_ylim([ymin, ymax])
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.ig_values*1e3, 'r-')
        ax2.set_ylabel('Ig (mA)', color='r')
        plt.xlim([xmin, xmax])
        plt.title('Turn-on waveform')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.show()

    def figure_qg(self):
        xmin = self.t1 - 30e-9
        xmax = self.t2 + 100e-9

        xmin_idx, xmax_idx = self._get_idx(xmin, xmax)
        time = self.time[xmin_idx:xmax_idx]
        ig = self.ig_values[xmin_idx:xmax_idx]
        qg_values = np.zeros_like(time)
        for i in range(len(time)):
            qg_values[i] = np.trapz(ig[:i+1], time[:i+1])

        ymin, ymax = self._autoscale_y(time, qg_values*1e9, xmin, xmax, margin=0.15)
        plt.plot(time, qg_values*1e9)
        plt.vlines(self.t1, ymin, ymax, color='green')
        plt.vlines(self.t2, ymin, ymax, color='green')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.xlabel('Time (ns)')
        plt.ylabel('Qg (nC)')
        plt.grid()
        plt.title('Qgoff')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.show()

        xmin = self.t3 - 30e-9
        xmax = self.t4 + 100e-9

        xmin_idx, xmax_idx = self._get_idx(xmin, xmax)
        time = self.time[xmin_idx:xmax_idx]
        ig = self.ig_values[xmin_idx:xmax_idx]
        qg_values = np.zeros_like(time)
        for i in range(len(time)):
            qg_values[i] = np.trapz(ig[:i+1], time[:i+1])

        ymin, ymax = self._autoscale_y(time, qg_values*1e9, xmin, xmax, margin=0.15)
        plt.plot(time, qg_values*1e9)
        plt.vlines(self.t3, ymin, ymax, color='green')
        plt.vlines(self.t4, ymin, ymax, color='green')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.xlabel('Time (ns)')
        plt.ylabel('Qg (nC)')
        plt.grid()
        plt.title('Qgon')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(self._func_time))
        plt.show()
