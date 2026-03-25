"""
Tektronix WFM file reader based on download from Tektronix
Refactored by Z. Ansari to change returned data structures, make PEP8 compliant.

Original comments:
wfm reader proof-of-concept
https://www.tek.com/sample-license
Reads volts vs. time records (including fastframes) from little-endian
version 3 WFM files
See Also
Performance Oscilloscope Reference Waveform File Format
Tektronix part # 077-0220-10
https://www.tek.com/oscilloscope/dpo7000-digital-phosphor-oscilloscope-manual-4
"""

from struct import unpack_from
import numpy as np


class WfmReadError(Exception):
    """error for unexpected things"""

    pass


def wfm_from_bytes(buffer):
    """return sample data from target WFM file"""
    f = buffer
    hbytes = f[:838]
    meta = decode_header(hbytes)
    # file signature checks
    if meta["byte_order"] != 0x0F0F:
        raise WfmReadError("big-endian not supported in this example")
    if meta["version"] != b":WFM#003":
        raise WfmReadError("only version 3 wfms supported in this example")
    if meta["imp_dim_count"] != 1:
        raise WfmReadError("imp dim count not 1")
    if meta["exp_dim_count"] != 1:
        raise WfmReadError("exp dim count not 1")
    if meta["record_type"] != 2:
        raise WfmReadError("not WFMDATA_VECTOR")
    if meta["exp_dim_1_type"] != 0:
        raise WfmReadError("not EXPLICIT_SAMPLE")
    if meta["time_base_1"] != 0:
        raise WfmReadError("not BASE_TIME")
    values = read_values(f, meta)
    trigger_times = extract_trigger_times(f, meta)
    wfm = dict(
        units=meta["units"],
        tstart=meta["tstart"],
        tstep=meta["tstep"],
        samples=meta["samples"],
        frame_count=meta["frame_count"],
        values=values,
        frame_times=trigger_times,
    )
    return wfm


def unpack(fmt, buffer, offset):
    """Return the first item in the struct.unpack_from method.
    This is just to improve legibility of the code."""
    return unpack_from(fmt, buffer, offset=offset)[0]


def decode_header(header_bytes):
    """returns a dict of wfm metadata"""
    wfm_info = {}
    if len(header_bytes) != 838:
        raise WfmReadError("wfm header bytes not 838")
    wfm_info["byte_order"] = unpack("H", header_bytes, 0)
    wfm_info["version"] = unpack("8s", header_bytes, 2)
    wfm_info["imp_dim_count"] = unpack("I", header_bytes, 114)
    wfm_info["exp_dim_count"] = unpack("I", header_bytes, 118)
    wfm_info["record_type"] = unpack("I", header_bytes, 122)
    wfm_info["exp_dim_1_type"] = unpack("I", header_bytes, 244)
    wfm_info["time_base_1"] = unpack("I", header_bytes, 768)
    wfm_info["fastframe"] = unpack("I", header_bytes, 78)
    wfm_info["wfm_count"] = unpack("I", header_bytes, 82)
    wfm_info["fast_frames_req"] = unpack("I", header_bytes, 146)
    wfm_info["fast_frames_acq"] = unpack("I", header_bytes, 150)
    wfm_info["summary_frame"] = unpack("h", header_bytes, 154)
    wfm_info["curve_offset"] = unpack("i", header_bytes, 16)
    # Ignore the summary frame if it exists
    nframes = unpack("I", header_bytes, 72) + 1
    if wfm_info["summary_frame"] == 1:
        wfm_info["frame_count"] = nframes - 1
    else:
        wfm_info["frame_count"] = nframes
    # curves_offset = 838 + ((frames - 1) * 54)
    # scaling factors
    wfm_info["vscale"] = unpack("d", header_bytes, 168)
    wfm_info["voffset"] = unpack("d", header_bytes, 176)
    wfm_info["tstart"] = unpack("d", header_bytes, 496)
    wfm_info["tstep"] = unpack("d", header_bytes, 488)
    # trigger detail
    wfm_info["gmt_fracsec"] = unpack("d", header_bytes, 796)  # frame index 0
    wfm_info["gmt_sec"] = unpack("I", header_bytes, 804)  # frame index 0
    # User data
    wfm_info["label"] = (unpack("32s", header_bytes, 40).decode("utf-8").split("\x00"))[
        0
    ]
    wfm_info["units"] = (
        unpack("20s", header_bytes, 276).decode("utf-8").split("\x00")
    )[0]
    # data offsets
    # frames are same size, only first frame offsets are used
    dpre = unpack("I", header_bytes, 822)
    wfm_info["dpre"] = dpre
    dpost = unpack("I", header_bytes, 826)
    wfm_info["dpost"] = dpost
    readbytes = dpost - dpre
    wfm_info["readbytes"] = readbytes
    allbytes = unpack("I", header_bytes, 830)
    wfm_info["allbytes"] = allbytes
    # sample data type detection
    code = unpack("i", header_bytes, 240)
    wfm_info["code"] = code
    bps = unpack("b", header_bytes, 15)  # bytes-per-sample
    wfm_info["bps"] = bps
    if code == 7 and bps == 1:
        dformat = "int8"
        samples = readbytes
    elif code == 0 and bps == 2:
        dformat = "int16"
        samples = readbytes // 2
    elif code == 4 and bps == 4:
        dformat = "single"
        samples = readbytes // 4
    else:
        raise WfmReadError("data type code or bytes-per-sample not understood")
    wfm_info["dformat"] = dformat
    wfm_info["samples"] = samples
    wfm_info["pts_per_frame"] = allbytes // bps
    wfm_info["pre_values"] = dpre // bps
    wfm_info["post_values"] = (allbytes - dpost) // bps
    return wfm_info


def read_values(buffer, meta):
    """Reads the whole array of data values and returns a 2D array of volts
    values by frame. Each frame consists of a number of buffer values before
    and after the stored sample values.
    """
    # Define the datatype for each frame
    nframes = meta["frame_count"]
    offset = meta["curve_offset"]
    pre_vals = meta["pre_values"]
    post_vals = meta["post_values"]
    dtype = meta["dformat"]
    vscale = meta["vscale"]
    voffset = meta["voffset"]
    samples = meta["samples"]
    frame_dtype = np.dtype(
        [
            ("pre", dtype, (pre_vals,)),
            ("volts", dtype, (samples,)),
            ("post", dtype, (post_vals,)),
        ]
    )
    # Read all frames as a structured array with keys 'pre', 'volts' and 'post'
    unscaled = np.frombuffer(buffer, dtype=frame_dtype, offset=offset, count=nframes)
    return unscaled["volts"] * vscale + voffset


def extract_trigger_times(buffer, meta):
    """Reads the fast frames metadata and returns a dict with the time
    of each trigger and the delta relative to the first frame."""
    # Define the datatype for each fastframe data object
    nframes = meta["frame_count"]
    # Initialise arrays for the GMT second and fraction for each trigger
    gmt_fracsec_array = np.zeros(meta["frame_count"], dtype=np.double)
    gmt_sec_array = np.zeros(meta["frame_count"], dtype=np.int32)
    # First value of each trigger is set from the header metadata
    gmt_zero = meta["gmt_sec"]
    gmt_fracsec = meta["gmt_fracsec"]
    gmt_sec_array[0] = gmt_zero
    gmt_fracsec_array[0] = gmt_fracsec
    tzero = gmt_zero + gmt_fracsec
    # Read the time offsets for remaining n-1 following fast frames
    ffstart = 838
    ff_dtype = np.dtype(
        [
            ("n", "i4"),
            ("tstep_offset", "f8"),
            ("frac_sec", "f8"),
            ("gmt_sec", "i4"),
        ]
    )
    # Read all frames as a structured array with keys 'pre', 'volts' and 'post'
    ffstart = 838
    fastframe_table = np.frombuffer(
        buffer, offset=ffstart, dtype=ff_dtype, count=nframes - 1
    )
    gmt_sec_array[1:] = fastframe_table["gmt_sec"]
    gmt_fracsec_array[1:] = fastframe_table["frac_sec"]
    # Initialise a dictionary with trigger times and delta relative to first
    trigger_times = {
        "gmt_sec": gmt_sec_array.copy(),
        "gmt_fracsec": gmt_fracsec_array.copy(),
        "delta": gmt_sec_array + gmt_fracsec_array - tzero,
    }
    return trigger_times
