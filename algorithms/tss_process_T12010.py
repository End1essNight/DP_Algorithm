import numpy as np
from .tss_file import TSSFile
from .wfm_edge import WFM_edge


class TSS_process:
    """Read the tss file."""
    def __init__(self, folder, tss_file):

        self.tss = TSSFile(folder, tss_file, source='Local')
        self.channel_labels = self.tss.channel_labels

        self._set_channels()
        self.time = self.tss.waveforms[self.vd_channel].time_for_frame()
        self.frame_n = self.tss.waveforms[self.vd_channel].frame_count

    def _set_channels(self):
        signal_groups = {
            "vd": ['Vd', 'Vds', 'Vmid'],
            "vg": ['Vg', 'Vgs', 'VgLS', 'Vg1_LS', 'Vg_LS'],
            "is": ['Is', 'Ids', 'Id', 'IsR'],
            'mc_gate': ['MC_GATE', 'V_MC_Gate'],
        }

        self.vd_channel = self.vg_channel = self.is_channel  = None
        self.mc_gate_channel = None

        for key, value in self.channel_labels.items():
            for attr, labels in signal_groups.items():
                if value in labels:
                    setattr(self, f"{attr}_channel", key)
                    break

    def T12010_extraction(self):
        peak_current = []
        peak_power = []
        dvdt_avg = []
        dvdt_max = []
        eoff_all = []
        for i in range(self.frame_n):
            try:
                is_values = self.tss.waveforms[self.is_channel].values_for_frame(i)
                vd = self.tss.waveforms[self.vd_channel].values_for_frame(i)
                vd_edges = WFM_edge(vd, self.time, falling_edge_number=0, rising_edge_number=1)
                
                idx_off_1 = vd_edges.get_edge(1, 'rising', 10)['thresh_idx']
                idx_off_2 = vd_edges.get_edge(1, 'rising', 90)['thresh_idx']
                off_1_idx = vd_edges.get_edge(1, 'rising', 1)['thresh_idx']

                time_turn_off = self.time[idx_off_1:idx_off_2]
                vd_turn_off = vd[idx_off_1:idx_off_2]
                dvdt_off = np.gradient(vd_turn_off, time_turn_off)
                dvdt_off_average = (vd[idx_off_1] - vd[idx_off_2]) / (self.time[idx_off_1] - self.time[idx_off_2])*1e-9
                dvdt_off_peak = max(dvdt_off)*1e-9
                
                power = np.multiply(vd, is_values)

                off_1_idx = vd_edges.get_edge(1, 'rising', 1)['thresh_idx']
                idx = vd_edges.get_edge(1, 'rising', 80)['thresh_idx']
                Is = is_values[idx]
                while Is <= 0:
                    idx += 1
                    Is = is_values[idx]

                while Is > 0:
                    idx += 1
                    Is = is_values[idx]
                off_2_idx = idx
                eoff = np.trapz(power[off_1_idx:off_2_idx], self.time[off_1_idx:off_2_idx])*1e6
                eoff_all.append(eoff)
                peak_current.append(max(is_values))
                peak_power.append(max(power))
                dvdt_avg.append(dvdt_off_average)
                dvdt_max.append(dvdt_off_peak)
            except Exception as e:
                print(f'frame {i}')
                print(f'frame {e}')
                peak_current.append(0)
                peak_power.append(0)
                dvdt_avg.append(0)
                dvdt_max.append(0)
                eoff_all.append(0)
        return peak_current, peak_power, dvdt_avg, dvdt_max, eoff_all

    def _get_idx(self, xmin, xmax):
        '''Get the closet index for xmin and xmax in time array.'''
        xmin_idx = np.argmin(np.abs(self.time - xmin))
        xmax_idx = np.argmin(np.abs(self.time - xmax))
        return xmin_idx, xmax_idx
