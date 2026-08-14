import sys
from pathlib import Path

project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))

from algorithms.tss_file import TSSFile
from algorithms.wfm_edge import WFM_edge
import numpy as np


def _get_idx(time, xmin, xmax):
    xmin_idx = np.argmin(np.abs(time - xmin))
    xmax_idx = np.argmin(np.abs(time - xmax))
    return xmin_idx, xmax_idx


def vd_overshoot_extraction(folder, file, pulse_number):
    tss = TSSFile(folder, file, source='Local')
    # channel_labels = tss.channel_labels

    time = tss.waveforms['ch1'].time_for_frame()
    vd = tss.waveforms['ch3'].values_for_frame(0)
    il = tss.waveforms['ch4'].values_for_frame(0)
    isv = tss.waveforms['ch6'].values_for_frame(0)
    vg = tss.waveforms['ch1'].values_for_frame(0)
    vgi = tss.waveforms['ch2'].values_for_frame(0)

    vd_edges = WFM_edge(vd, time, falling_edge_number=pulse_number,
                                rising_edge_number=pulse_number, 
                                peak_width=1,
                                peak_distance=10, 
                                sigma=100,
                                margin_ratio=3,)

    is_edges = WFM_edge(isv, time, falling_edge_number=pulse_number,
                            rising_edge_number=pulse_number,
                            peak_width=1,
                            peak_distance=10, 
                            sigma=100,
                            margin_ratio=3,)

    vg_edges = WFM_edge(vg, time, falling_edge_number=pulse_number,
                            rising_edge_number=pulse_number, 
                            peak_width=1,
                            peak_distance=10, 
                            sigma=500,
                            margin_ratio=10,)

    data = []
    for i in range(1, pulse_number+1):
        t1 = vd_edges.get_edge(i, 'rising', 90)['time']
        t2 = t1 + 1000e-9
        xmin_idx, xmax_idx = _get_idx(time, t1, t2)
        vd_overshoot = max(vd[xmin_idx:xmax_idx])
        il_value = np.average(il[xmax_idx:xmax_idx+10])

        idx_off_1 = is_edges.get_edge(i, 'falling', 90)['thresh_idx']
        idx_off_2 = is_edges.get_edge(i, 'falling', 10)['thresh_idx']
        idx_off2_1 = is_edges.get_edge(i, 'falling', 99)['thresh_idx']
        idx_off2_2 = is_edges.get_edge(i, 'falling', 1)['thresh_idx']
        time_turn_off = time[idx_off2_1:idx_off2_2]
        is_turn_off = isv[idx_off2_1:idx_off2_2]
        didt_off = np.gradient(is_turn_off, time_turn_off)
        didt_off_peak = min(didt_off)
        didt_off_average = (isv[idx_off_1] - isv[idx_off_2]) / (time[idx_off_1] - time[idx_off_2])

        t_vg_fall_idx = vg_edges.get_edge(i, 'falling', 99)['thresh_idx']
        vgi_max = max(vgi[t_vg_fall_idx-100:t_vg_fall_idx+10])
        time1 = time[t_vg_fall_idx]
        vgiv = vgi[t_vg_fall_idx]
        while vgiv >= 0:
            t_vg_fall_idx += 1
            vgiv = vgi[t_vg_fall_idx]
        delta_t = time[t_vg_fall_idx] - time1
        data.append([i, il_value, vd_overshoot, didt_off_peak*1e-9, didt_off_average*1e-9, vgi_max, delta_t*1e9])

    return data
