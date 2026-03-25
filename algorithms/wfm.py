"""
Class to model Tektronix wfm files.
Uses the refactored tekwfm.py downloaded from Tektronix.
"""

import numpy as np
from .tekwfm import wfm_from_bytes  # From Tek


class WFM:
    '''
    Reads the .wfm binary structure
    for analysis without saving to large
    files
    '''

    def __init__(self, file, channel, label, buffer):
        self.file = file
        self.channel = channel
        self.label = label
        self.samples = 0
        self.frame_count = 0
        self.values = np.array([])
        self.frame_times = dict()
        self.tstart = 0
        self.tstep = 1.0e-9
        self.tzero = 0.0 # Time offset
        wfm_data = wfm_from_bytes(buffer)
        for key, value in wfm_data.items():
            self.__setattr__(key, value)

    def values_for_window(self, tstart, twidth):
        """Provides a subset of data for all frames for the time window
        defined by tstart and tstop"""
        time = self.time_for_frame()
        tstop = tstart + twidth
        indices = np.where((time >= tstart) & (time < tstop))[0]
        return self.values[:,indices]

    def mean_values_for_window(self, tstart, twidth):
        """Returns a mean of the values for the window by frame"""
        values = self.values_for_window(tstart, twidth)
        return values.mean(axis=1)

    def time_and_values_for_frame(self, frame, max_points=None):
        """Returns time and volts numpy arrays.
        If max_points is provided then the traces are downsampled to limit the
        number of points in the array"""
        if max_points is None:
            downsample = 1
        else:
            downsample = max(1, self.samples // max_points)
        values = self.values_for_frame(frame)[::downsample]
        time = self.time_for_frame()[::downsample]
        return time, values

    def time_for_frame(self):
        """Returns an array of times."""
        points = self.samples
        tstart = self.tstart
        tstep = self.tstep
        tstop = points * tstep + tstart
        time = np.linspace(tstart, tstop, points, endpoint=False)
        return time - self.tzero

    def values_for_frame(self, frame):
        """Returns an array of times"""
        return self.values[frame]
