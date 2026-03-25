import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def find_edges(smoothed, time, falling_edge_n, rising_edge_n, peak_width, peak_distance, margin_ratio):
    """Finds the edges in a 1D array of a first derivative of Sobel filter.
    Returns a dict of edge indices and edge direction (rising or falling)"""
    derivative = np.gradient(smoothed, time)
    prominence = np.abs(np.abs(derivative)).max() / 8
    width = peak_width  # The minimum width of a peak
    distance = peak_distance  # The minimum data points between peaks
    indices, props = find_peaks(np.abs(derivative), prominence=prominence, width=width, distance=distance)
    results = []
    for i, idx in enumerate(indices):
        try:
            width = props["widths"][i]
            margin = margin_ratio * int(width)
            left = idx - margin
            if left < 0:
                left = 0
            right = idx + margin
            if right >= len(derivative):
                right = len(derivative) - 1
            value = derivative[idx]
            direction = "rising" if value > 0 else "falling"
            this_peak = {
                "peak_idx": idx,
                "direction": direction,
                "width": width,
                "left": left,
                "right": right,
                "derivative": value
            }
        except Exception:
            # Return a dict with default values
            this_peak = {
                "peak_idx": idx,
                "direction": direction,
                "width": 0,
                "left": 0,
                "right": 1,
                "derivative": value
            }
        results.append(this_peak)

    n_fall = len([x for x in results if x['direction'] == 'falling'])
    n_rise = len([x for x in results if x['direction'] == 'rising'])

    # if n_fall < falling_edge_n or n_rise < rising_edge_n:
    # raise Exception("Cannot detect enough pulse edges!")

    if n_fall > falling_edge_n or n_rise > rising_edge_n:
        # Sort the results based on the absolute value of the 'derivative' key
        sorted_results = sorted(results, key=lambda x: abs(x['derivative']), reverse=True)

        # Filter the sorted results to keep only the first two elements with positive 'derivative' and the first two elements with negative 'derivative'
        filtered_results = [res for res in sorted_results if res['derivative'] > 0][:rising_edge_n] + [res for res in sorted_results if res['derivative'] < 0][:falling_edge_n]
        results = sorted(filtered_results, key=lambda x: x['peak_idx'])

    n_fall = len([x for x in results if x['direction'] == 'falling'])
    n_rise = len([x for x in results if x['direction'] == 'rising'])

    return results, n_fall, n_rise


class WFM_edge:

    def __init__(self, wfm, time, falling_edge_number=2, rising_edge_number=2, flag_uneven_plateau=False
                 , peak_width=1, peak_distance=10, sigma=25, margin_ratio=20):
        self.wfm = wfm
        self.time = time
        self.edges = None
        self.fup = flag_uneven_plateau
        self.falling_edge_number = falling_edge_number
        self.rising_edge_number = rising_edge_number
        self._find_edges_and_levels(peak_width, peak_distance, sigma, margin_ratio)

    def get_edge(self, edge_num, direction, threshold):
        """Return a list of dicts of extracted threshold time"""
        # self._find_edges_and_levels()

        edge = self.edge_data(edge_num, direction)
        if edge is not None:
            value_threshold = self._pct_threshold(edge, threshold)
            edge_idx = edge["peak_idx"]

            left = edge["left"]
            right = edge["right"]
            thresh_idx = self._idx_for_value(edge_idx, value_threshold, left, right)

            if direction == 'falling':
                if self.wfm[thresh_idx] >= value_threshold:
                    thresh_time = np.interp(value_threshold, [self.wfm[thresh_idx + 1], self.wfm[thresh_idx]], [self.time[thresh_idx + 1], self.time[thresh_idx]])
                else:
                    thresh_time = np.interp(value_threshold, [self.wfm[thresh_idx], self.wfm[thresh_idx - 1]], [self.time[thresh_idx], self.time[thresh_idx - 1]])
            if direction == 'rising':
                if self.wfm[thresh_idx] >= value_threshold:
                    thresh_time = np.interp(value_threshold, [self.wfm[thresh_idx - 1], self.wfm[thresh_idx]], [self.time[thresh_idx - 1], self.time[thresh_idx]])
                else:
                    thresh_time = np.interp(value_threshold, [self.wfm[thresh_idx], self.wfm[thresh_idx + 1]], [self.time[thresh_idx], self.time[thresh_idx + 1]])

            result = {
                "edge": edge_num,
                "direction": direction,
                "threshold": threshold,
                "thresh_value": value_threshold,
                "thresh_idx": thresh_idx,
                "time": thresh_time,
                }
            result.update(edge)
        else:
            # Return a dict with default values
            result = {
                "peak_idx": 0,
                "direction": direction,
                "width": 0,
                "left": 0,
                "right": 1,
                "lo": 0,
                "hi": 0,
                "edge": 0,
                "direction": direction,
                "threshold": threshold,
                "thresh_value": 0,
                "thresh_idx": 0,
                "time": 0,
            }

        return result

    def _find_edges_and_levels(self, peak_width, peak_distance, sigma, margin_ratio):
        """Find the edges and then use the left and right ends of edges
        to calculate the low and high levels.
        Uses the smoothed array to calculate low and high."""
        self.smoothed = gaussian_filter1d(self.wfm, sigma=sigma)
        edges, n_fall, n_rise = find_edges(self.smoothed, self.time, self.falling_edge_number, self.rising_edge_number
                                           , peak_width, peak_distance, margin_ratio)

        for i, edge in enumerate(edges):
            left = edge["left"]
            right = edge["right"]
            direction = edge["direction"]
            if direction == "falling":
                edge["lo"] = min(self.smoothed[left:right])
                if self.fup:
                    edge["hi"] = max(self.wfm[left:right])
                else:
                    edge["hi"] = max(self.smoothed[left:right])
            if direction == "rising":
                edge["hi"] = max(self.smoothed[left:right])
                if self.fup:
                    edge["lo"] = min(self.wfm[left:right])
                else:
                    edge["lo"] = min(self.smoothed[left:right])

        self.falling_edge_number = n_fall
        self.rising_edge_number = n_rise
        self._edges = edges

    def edge_data(self, edge_num, direction):
        """Returns parameters for a specific edge"""
        if direction not in ("rising", "falling"):
            raise ValueError("Illegal value for direction")
        edges = [x for x in self._edges if x["direction"] == direction]
        edge_num = edge_num % len(edges)
        return edges[edge_num - 1]

    def _pct_threshold(self, edge, percent):
        """Returns the value relative to high and low values for
        the percent of peak-to-peak above the low level"""
        if edge is not None:
            lo = edge["lo"]
            hi = edge["hi"]
            return lo + percent / 100 * (hi - lo)
        else:
            return 0

    def _idx_for_value(self, edge_idx, thresh, left, right):
        """Returns the index for the closest value in values"""
        crossing_indices = np.where(np.diff(np.sign(self.wfm[left:right] - thresh)))[0] + left
        if len(crossing_indices) > 0:
            desired_time_idx = crossing_indices[np.argmin(np.abs(crossing_indices - edge_idx))]
        else:
            desired_time_idx = np.argmin(np.abs(self.wfm[left:right] - thresh)) + left

        return desired_time_idx
