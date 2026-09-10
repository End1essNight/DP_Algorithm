import sys
from pathlib import Path

project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))

from algorithms.tss_file import TSSFile
from algorithms.wfm_edge import WFM_edge
import numpy as np

def desat_extract(folder, file):
    tss = TSSFile(folder, file, source='Local')
    # channel_labels = tss.channel_labels

    time = tss.waveforms['ch1'].time_for_frame()
    vg = tss.waveforms['ch1'].values_for_frame(0)
    vgi = tss.waveforms['ch8'].values_for_frame(0)
    # vd = tss.waveforms['ch3'].values_for_frame(0)
    Is = tss.waveforms['ch2'].values_for_frame(0)
    ig = tss.waveforms['ch7'].values_for_frame(0)
    vblnk = tss.waveforms['ch6'].values_for_frame(0)
    vcl = tss.waveforms['ch4'].values_for_frame(0)
    flt = tss.waveforms['ch5'].values_for_frame(0)

    vg_edges = WFM_edge(vg, time, sigma=100)
    vblnk_edges = WFM_edge(vblnk, time, sigma=200)
    time1 = vg_edges.get_edge(2, 'rising', 10)['time']
    time2 = vblnk_edges.get_edge(2, 'falling', 90)['time']
    desat_blank = (time2 - time1) * 1e9

    flt_edges = WFM_edge(flt, time, sigma=100)
    time2 = vblnk_edges.get_edge(2, 'falling', 90)['time']
    time3 = flt_edges.get_edge(2, 'falling', 90)['time']
    desat_t_react = (time3 - time2) * 1e9

    time4 = vg_edges.get_edge(1, 'falling', 10)['time']
    time5 = flt_edges.get_edge(1, 'rising', 10)['time']
    desat_reset_time = (time5 - time4) * 1e9

    vgi_edges = WFM_edge(vgi, time, sigma=100)
    time6 = vgi_edges.get_edge(1, 'falling', 90)['time']
    time7 = vgi_edges.get_edge(1, 'falling', 10)['time']
    desat_VGI_tlow = (time7 - time6) * 1e9

    idx1 = vgi_edges.get_edge(1, 'falling', 99)['thresh_idx']
    idx2 = vgi_edges.get_edge(1, 'falling', 1)['thresh_idx']
    is_max = max(Is[(idx1-(idx2-idx1)):(idx2+(idx2-idx1))])

    time_inter = time[1] - time[0]
    idx4 = vg_edges.get_edge(1, 'falling', 99)['thresh_idx'] - int(0.1e-6/time_inter)
    idx3 = np.argmin(np.abs(time - (time[idx4]-0.1e-6)))

    desat_Ig_max = max(ig[idx3:idx4])*1e3
    desat_flt_Vactive = np.average(flt[idx3:idx4])*1e3

    time8 = flt_edges.get_edge(1, 'falling', 10)['time'] - desat_t_react*1e-9
    idx5 = np.argmin(np.abs(time - time8))
    desat_VDS_th = np.average(vcl[idx5-3:idx5+3])

    return [
        desat_blank,
        desat_t_react,
        desat_reset_time,
        desat_VGI_tlow,
        desat_Ig_max,
        desat_VDS_th,
        desat_flt_Vactive,
        is_max,
    ]
