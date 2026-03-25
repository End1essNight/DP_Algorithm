import os
import pandas as pd
import json
import sys
from pathlib import Path

project_root = Path.cwd().parents[0]
sys.path.append(str(project_root))

from algorithms.tss_file import TSSFile
from algorithms.wfm_edge import WFM_edge
import numpy as np

def desat_extract(folder, file):
    tss = TSSFile(folder, file, source='Local')
    channel_labels = tss.channel_labels
    print(channel_labels)
    time = tss.waveforms['ch1'].time_for_frame()
    vg = tss.waveforms['ch1'].values_for_frame(0)
    vgi = tss.waveforms['ch2'].values_for_frame(0)
    # vd = tss.waveforms['ch3'].values_for_frame(0)
    Is = tss.waveforms['ch4'].values_for_frame(0)
    ig = tss.waveforms['ch5'].values_for_frame(0)
    vblnk = tss.waveforms['ch6'].values_for_frame(0)
    vcl = tss.waveforms['ch7'].values_for_frame(0)
    flt = tss.waveforms['ch8'].values_for_frame(0)

    vg_edges = WFM_edge(vg, time, sigma=80)
    flt_edges = WFM_edge(flt, time, sigma=80)
    vblnk_edges = WFM_edge(vblnk, time, sigma=80)
    vgi_edges = WFM_edge(vgi, time, sigma=80)

    time1 = vg_edges.get_edge(1, 'rising', 10)['time']
    time2 = vblnk_edges.get_edge(1, 'falling', 90)['time']
    desat_blank = (time2 - time1) * 1e9

    time2 = vblnk_edges.get_edge(2, 'falling', 90)['time']
    time3 = flt_edges.get_edge(2, 'falling', 90)['time']
    desat_t_react = (time3 - time2) * 1e9

    time4 = vg_edges.get_edge(1, 'falling', 10)['time']
    time5 = flt_edges.get_edge(1, 'rising', 10)['time']
    desat_reset_time = (time5 - time4) * 1e9

    time6 = vgi_edges.get_edge(1, 'falling', 90)['time']
    time7 = vgi_edges.get_edge(1, 'falling', 10)['time']
    desat_VGI_tlow = (time7 - time6) * 1e9

    idx1 = vgi_edges.get_edge(1, 'falling', 99)['thresh_idx']
    idx2 = vgi_edges.get_edge(1, 'falling', 1)['thresh_idx']
    is_max = max(Is[(idx1-(idx2-idx1)):(idx2+(idx2-idx1))])

    time_inter = time[1] - time[0]
    idx4 = vg_edges.get_edge(1, 'falling', 99)['thresh_idx'] - int(0.1e-6/time_inter)
    idx3 = np.argmin(np.abs(time - (time[idx4]-2e-6)))

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


def desat_extract_TOLL(folder, file):
    tss = TSSFile(folder, file, source='Local')
    channel_labels = tss.channel_labels
    channels = _set_channels(channel_labels)
    time = tss.waveforms[channels['vd_channel']].time_for_frame()
    vd = tss.waveforms[channels['vd_channel']].values_for_frame(0)
    Is = tss.waveforms[channels['is_channel']].values_for_frame(0)
    vcl = tss.waveforms[channels['vcl_channel']].values_for_frame(0)

    is_edges = WFM_edge(Is, time, falling_edge_number=2, rising_edge_number=1, sigma=100, margin_ratio=3)
    # time1 = is_edges.get_edge(1, 'falling', 90)['time']
    time2 = is_edges.get_edge(1, 'rising', 10)['time']
    time3 = is_edges.get_edge(2, 'falling', 90)['time']
    idx1 = is_edges.get_edge(1, 'falling', 90)['thresh_idx']
    idx2 = is_edges.get_edge(1, 'rising', 10)['thresh_idx']
    idx3 = is_edges.get_edge(2, 'falling', 90)['thresh_idx']

    blanking_react = (time3 - time2) * 1e9
    idx = idx1 - (idx3 - idx2)
    desat_is = np.average(Is[idx-3:idx+3])
    desat_vcl = np.average(vcl[idx-3:idx+3])
    desat_vd = np.average(vd[idx-3:idx+3])

    return [blanking_react, desat_is, desat_vcl, desat_vd]


def _set_channels(channel_labels):
    signal_groups = {
        "vd": ['Vd', 'Vds', 'Vmid'],
        "vg": ['Vg', 'Vgs', 'VgLS', 'Vg1_LS', 'Vg_LS'],
        "is": ['Is', 'Ids', 'Id', 'IsR'],
        "il": ['IL'],
        "vdc": ['DCLink'],
        "vcl": ['Vcl'],
        "ig": ['Ig'],
        'mc_gate': ['MC_GATE'],
        "vgi": ['Vgi'],
    }

    channels = {}

    for key, value in channel_labels.items():
        for attr, labels in signal_groups.items():
            if value in labels:
                channels[f"{attr}_channel"] = key
                break

    return channels

def find_file(filename, search_dir):
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return root
    return None

def glob_files(base_path, file_type):
    paths = Path(base_path).rglob(file_type)
    parent_folders = []
    file_names = []

    for p in paths:
        parent_folders.append(p.parent)
        file_names.append(p.name)

    return parent_folders, file_names

def desat_process(input_folders, output_folder, output_name, comment_contains):
    output = [
        ['Mark ID', 'Time stamp', 'Comment', 'L (uH)', 'Vdc (V)', 'Ipeak (A)', 'inter_pulse_delay (ns)', 'pulse2_width (ns)', "dead_time (ns)",
         "gate_voltage", 'gate_resistor', 'Vd (V)', 'Id (A)', 'vdd', 'error',
         'DESAT_blank', 'DESAT_t_react', 'DESAT_reset_time', 'DESAT_VGI_tlow', 'DESAT_Ig_max', 'DESAT_VDS_th', 'DESAT_flt_Vactive', 'Is_MAX']
    ]

    for input_folder in input_folders:
        json_folders, json_files = glob_files(input_folder, '*.json')
        for i in range(len(json_files)):
            json_file_path = os.path.join(json_folders[i], json_files[i])
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)
            time_stamp = json_files[i][3:18]
            tss_file = 'Tek' + time_stamp + '.tss'
            tss_folder = find_file(tss_file, json_folders[i])
            if tss_folder is None:
                continue

            chip_id = json_data['chip_id']
            vdp = json_data['dc_voltage']
            idp = json_data['peak_current']
            induc = json_data['inductance']
            comment = json_data['comment']
            inter_pulse_delay = json_data['inter_pulse_delay']
            pulse2_width = json_data['pulse2_width']
            dead_time = json_data['dead_time']
            gate_voltage = json_data['gate_voltage']
            gate_resistor = json_data['gate_resistor']
            vdd = json_data['vdd']
            vd = json_data['drain_voltage']
            id = json_data['drain_current']
            """if int(vd) != 400 or int(id) != 350:
                continue"""

            if comment_contains not in comment:
                continue
            error = ''
            print(f'{tss_file} processing starts.')
            try:
                results = desat_extract(tss_folder, tss_file)
            except Exception as e:
                print(f'{tss_file}')
                print(e)
                results = [0 for _ in range(8)]
                error = 'error'

            line = [chip_id, time_stamp, comment, induc, vdp, idp, inter_pulse_delay,
                    pulse2_width, dead_time, gate_voltage, gate_resistor, vd, id, vdd, error,] + results
            output.append(line)

            print(f'{tss_file} processing is finished.')

    output_df = pd.DataFrame(output)
    output_path = os.path.join(output_folder, output_name + '.csv')
    output_df.to_csv(output_path, index=False, header=False)


def desat_TOLL_process(input_folders, output_folder, output_name, comment_contains):
    output = [
        ['Mark ID', 'Time stamp', 'Comment', 'L (uH)', 'Vdc (V)', 'Ipeak (A)', 'inter_pulse_delay (ns)', 'pulse2_width (ns)', "dead_time (ns)",
         "gate_voltage", 'gate_resistor', 'Vd (V)', 'Id (A)', 'vdd', 'error',
         'Blanking + React time (ns)', 'Is_DESAT (A)', 'Vcl_DESAT (V)', 'Vd_DESAT (V)',
         ]
    ]

    for input_folder in input_folders:
        json_folders, json_files = glob_files(input_folder, '*.json')
        for i in range(len(json_files)):
            json_file_path = os.path.join(json_folders[i], json_files[i])
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)
            time_stamp = json_files[i][3:18]
            tss_file = 'Tek' + time_stamp + '.tss'
            tss_folder = find_file(tss_file, json_folders[i])
            if tss_folder is None:
                continue

            chip_id = json_data['chip_id']
            vdp = json_data['dc_voltage']
            idp = json_data['peak_current']
            induc = json_data['inductance']
            comment = json_data['comment']
            inter_pulse_delay = json_data['inter_pulse_delay']
            pulse2_width = json_data['pulse2_width']
            dead_time = json_data['dead_time']
            gate_voltage = json_data['gate_voltage']
            gate_resistor = json_data['gate_resistor']
            vdd = json_data['vdd']
            vd = json_data['drain_voltage']
            id = json_data['drain_current']
            """if int(vd) != 400 or int(id) != 350:
                continue"""

            if comment_contains not in comment:
                continue
            error = ''
            # print(f'{tss_file} processing starts.')
            try:
                results = desat_extract_TOLL(tss_folder, tss_file)
            except Exception as e:
                print(f'{tss_file}')
                print(e)
                results = [0 for _ in range(4)]
                error = 'error'

            line = [chip_id, time_stamp, comment, induc, vdp, idp, inter_pulse_delay,
                    pulse2_width, dead_time, gate_voltage, gate_resistor, vd, id, vdd, error,] + results
            output.append(line)

            print(f'{tss_file} processing is finished.')

    output_df = pd.DataFrame(output)
    output_path = os.path.join(output_folder, output_name + '.csv')
    output_df.to_csv(output_path, index=False, header=False)
