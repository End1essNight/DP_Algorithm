from scipy.signal import savgol_filter
import numpy as np


def get_fall_time(wfm, time, thresh, edge, vd):
    t0 = edge['time']
    if thresh == 10:
        results = {
            'oscillation': False,
            'time': t0,
            'vd_wfm_smoothed': None,
        }
        return results

    idx1 = vd.get_edge(2, 'falling', thresh)['thresh_idx']
    idx2 = edge['thresh_idx']

    osc = oscillation(wfm[idx1:idx2])
    if osc:
        wfm_smoothed = wfm.copy()
        idx3 = idx1 + 200
        wfm_smoothed[idx1:idx3] = savgol_filter(wfm[idx1:idx3], window_length=20, polyorder=3)
        value_threshold = edge['thresh_value']
        left = edge['left']
        right = edge['right']
        edge_idx = edge["peak_idx"]

        thresh_idx = _idx_for_value(wfm_smoothed, edge_idx, value_threshold, left, right)
        thresh_time = find_thresh_time(wfm_smoothed, time, thresh_idx, value_threshold)
        results = {
            'oscillation': osc,
            'time': min(thresh_time, t0),
            'vd_wfm_smoothed': wfm_smoothed,
        }
    else:
        results = {
            'oscillation': osc,
            'time': t0,
            'vd_wfm_smoothed': None,
        }

    return results


def oscillation(arr):
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            return True

    return False


def find_thresh_time(wfm, time, thresh_idx, value_threshold):
    if wfm[thresh_idx] >= value_threshold:
        thresh_time = np.interp(value_threshold, [wfm[thresh_idx + 1], wfm[thresh_idx]], [time[thresh_idx + 1], time[thresh_idx]])
    else:
        thresh_time = np.interp(value_threshold, [wfm[thresh_idx], wfm[thresh_idx - 1]], [time[thresh_idx], time[thresh_idx - 1]])

    return thresh_time


def _idx_for_value(wfm, edge_idx, thresh, left, right):
    """Returns the index for the closest value in values"""
    crossing_indices = np.where(np.diff(np.sign(wfm[left:right] - thresh)))[0] + left
    if len(crossing_indices) > 0:
        desired_time_idx = crossing_indices[np.argmin(np.abs(crossing_indices - edge_idx))]
    else:
        desired_time_idx = np.argmin(np.abs(wfm[left:right] - thresh)) + left

    return desired_time_idx
