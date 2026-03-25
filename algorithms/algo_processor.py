from .tss_process import TSS_process
import os
import pandas as pd
import json
from pathlib import Path

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

def single_processor(folder, tss_file, config_file, item='time', 
                     auto_deskew=False, manual_deskew=False, deskew_time=0, lshunt=None):

    path = find_file(tss_file, folder)
    tss = TSS_process(path, tss_file, config_file)
    if item == 'energy':
        energy_output = tss.energy_extraction(auto_deskew=auto_deskew, manual_deskew=manual_deskew, deskew_time=deskew_time, lshunt=lshunt)
        print(f"Eon (uJ): {energy_output[0]}, Eoff (uJ): {energy_output[1]}, Deskew time (ns): {energy_output[2]}, Lloop: {energy_output[3]}")
        tss.figure_before_deskew()
        tss.figure_time_deskew()
        tss.figure_on_IV()
        tss.figure_off_IV()
        tss.figure_on_energy()
        tss.figure_off_energy()
    elif item == 'time':
        time_output = tss.time_extraction()
        print(f"tdon (ns): {time_output[0]}, tf (ns): {time_output[1]}, tdoff (ns): {time_output[2]}, tr (ns): {time_output[3]}")
        tss.figure_on_off_VV()
    elif item == 'qg':
        qg = tss.gate_charge_extraction()
        print(f'Qg_ON (nC): {qg[1]}, Qg_OFF (nC): {qg[0]}')
        tss.figure_gate_waveform()
        tss.figure_qg()
    elif item == 'P42016':
        # dvdt = tss.dvdt_extraction()
        # print(f'dvdt_on (V/ns): {dvdt[0]}, dvdt_off (V/ns): {dvdt[2]}')
        didt = tss.P42016_extraction()
        print(f'didt_on (A/ns): {didt[0]}, didt_off (A/ns): {didt[1]}, is_at_mcgate-1.5 (A): {didt[2]}')
        # dvdt = tss.dvdt_extraction(plot_figures=True)
        didt = tss.P42016_extraction(plot_figures=True)
        dvdt = tss.P42016_vgi_extraction()
        print(f'dVgi/dt_on (V/ns): {dvdt[0]}, dVgi/dt_off (V/ns): {dvdt[1]}')
        dvdt = tss.P42016_vgi_extraction(plot_figures=True)

def multi_processor(input_folders, output_folder, output_name, comment_contains, config_file):
    # Extract all parameter time, energy, qg, dvdt
    output = [
        ['Mark ID', 'Time stamp', 'Comment', 'L (uH)', 'Vd (V)', 'Id (A)', 'inter_pulse_delay (ns)', 'pulse2_width (ns)', "dead_time (ns)",
        "gate_voltage", 'gate_resistor', 'vdd', 'error', 'tdon (ns)', 'tf (ns)', 'tdoff (ns)', 'tr (ns)', 'Eon (uJ)', 'Eoff (uJ)',
        'Qgon (nC)', 'Qgoff (nC)',
        'Vd_dvdt_on_90_10 (V/ns)', 'Vd_dvdt_on_MAX (V/ns)', 'Vd_dvdt_off_10_90 (V/ns)', 'Vd_dvdt_off_MAX (V/ns)']
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
            # if idp != 30:
            #    continue
            # if vdd != 15:
            #    continue

            if comment_contains not in comment:
                continue
            tss = TSS_process(tss_folder, tss_file, config_file)

            error = ''
            try:
                energy_output = tss.energy_extraction()
                eon = energy_output[0]
                eoff = energy_output[1]
            except Exception as e:
                eon = 0
                eoff = 0
                error = 'error'

            try:
                time_output = tss.time_extraction()
                tdon = time_output[0]
                tf = time_output[1]
                tdoff = time_output[2]
                tr = time_output[3]
            except Exception as e:
                tdon = 0
                tf = 0
                tdoff = 0
                tr = 0
                error = 'error'

            try:
                qg_output = tss.gate_charge_extraction()
                qgoff = qg_output[0]
                qgon = qg_output[1]
            except Exception as e:
                qgoff = 0
                qgon = 0
                error = 'error'

            try:
                dvdt_output = tss.dvdt_extraction()
                dvdt_on_average = dvdt_output[0]
                dvdt_on_peak = dvdt_output[1]
                dvdt_off_average = dvdt_output[2]
                dvdt_off_peak = dvdt_output[3]
            except Exception as e:
                dvdt_on_average, dvdt_on_peak, dvdt_off_average, dvdt_off_peak = 0, 0, 0, 0
                error = 'error'

            line = [chip_id, time_stamp, comment, induc, vdp, idp, inter_pulse_delay,
                    pulse2_width, dead_time, gate_voltage, gate_resistor, vdd, error,
                    tdon, tf, tdoff, tr, eon, eoff,
                    qgon, qgoff,
                    dvdt_on_average, dvdt_on_peak, dvdt_off_average, dvdt_off_peak]
            output.append(line)

            print(f'{tss_file} processing is finished.')

    output_df = pd.DataFrame(output)
    output_path = os.path.join(output_folder, output_name + '.csv')
    output_df.to_csv(output_path, index=False, header=False)

def multi_processor_P42016(input_folders, output_folder, output_name, comment_contains, config_file):
    # Specific processor for Kaspars requests on P42016 dynamic data
    output = [
        ['Mark ID', 'Time stamp', 'Comment', 'L (uH)', 'Vd (V)', 'Id (A)', 'inter_pulse_delay (ns)', 'pulse2_width (ns)', "dead_time (ns)",
        "gate_voltage", 'gate_resistor', 'vdd', 'error', 'tdon (ns)', 'tf (ns)', 'tdoff (ns)', 'tr (ns)', 'Eon (uJ)', 'Eoff (uJ)', 'Qgon (nC)', 'Qgoff (nC)',
        'Vd_dvdt_on_9010 (V/ns)', 'Is_didt_on_1090 (A/ns)', 'Vd_dvdt_off_1090 (V/ns)', 'Is_didt_off_9010 (A/ns)', 'Is_at_MCGATE-1.5V (A)',]
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

            if comment_contains not in comment:
                continue
            tss = TSS_process(tss_folder, tss_file, config_file)

            error = ''
            try:
                energy_output = tss.energy_extraction()
                eon = energy_output[0]
                eoff = energy_output[1]
            except Exception as e:
                eon = 0
                eoff = 0
                error = 'error'

            try:
                time_output = tss.time_extraction()
                tdon = time_output[0]
                tf = time_output[1]
                tdoff = time_output[2]
                tr = time_output[3]
            except Exception as e:
                tdon = 0
                tf = 0
                tdoff = 0
                tr = 0
                error = 'error'

            try:
                qg_output = tss.gate_charge_extraction()
                qgoff = qg_output[0]
                qgon = qg_output[1]
            except Exception as e:
                qgoff = 0
                qgon = 0
                error = 'error'

            try:
                dvdt_output = tss.dvdt_extraction()
                dvdt_on_average = dvdt_output[0]
                dvdt_off_average = dvdt_output[2]
            except Exception as e:
                dvdt_on_average, dvdt_off_average = 0, 0
                error = 'error'

            try:
                dvdt_output = tss.P42016_extraction()
                didt_on_average = dvdt_output[0]
                didt_off_average = dvdt_output[1]
                is_at_mcgate = dvdt_output[2]
            except Exception as e:
                didt_on_average, didt_off_average, is_at_mcgate = 0, 0, 0
                error = 'error'

            line = [chip_id, time_stamp, comment, induc, vdp, idp, inter_pulse_delay,
                    pulse2_width, dead_time, gate_voltage, gate_resistor, vdd, error,
                    tdon, tf, tdoff, tr, eon, eoff, qgon, qgoff,
                    dvdt_on_average, didt_on_average, dvdt_off_average, didt_off_average, is_at_mcgate]
            output.append(line)

            print(f'{tss_file} processing is finished.')

    output_df = pd.DataFrame(output)
    output_path = os.path.join(output_folder, output_name + '.csv')
    output_df.to_csv(output_path, index=False, header=False)


def multi_processor_P42016_didt_vgi(input_folders, output_folder, output_name, comment_contains, config_file):
    # Specific processor for Kaspars requests on P42016 dynamic data
    output = [
        ['Mark ID', 'Time stamp', 'Comment', 'L (uH)', 'Vd (V)', 'Id (A)', 'inter_pulse_delay (ns)', 'pulse2_width (ns)', "dead_time (ns)",
        "gate_voltage", 'gate_resistor', 'vdd', 'error', 'dIs/dt_on_1090 (A/ns)', 'dIs/dt_off_9010 (A/ns)', 
        'dVgi/dt_on_1090 (V/ns)', 'dVgi/dt_off_9010 (V/ns)']
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

            if comment_contains not in comment:
                continue
            tss = TSS_process(tss_folder, tss_file, config_file)

            error = ''
            try:
                dvdt_output = tss.P42016_extraction()
                didt_on_average = dvdt_output[0]
                didt_off_average = dvdt_output[1]
            except Exception as e:
                didt_on_average, didt_off_average = 0, 0
                error = 'error'
            try:
                dvdt_output = tss.P42016_vgi_extraction()
                dvgidt_on_average = dvdt_output[0]
                dvgidt_off_average = dvdt_output[1]
            except Exception as e:
                dvgidt_on_average, dvgidt_off_average = 0, 0
                error = 'error'

            line = [chip_id, time_stamp, comment, induc, vdp, idp, inter_pulse_delay,
                    pulse2_width, dead_time, gate_voltage, gate_resistor, vdd, error,
                    didt_on_average, didt_off_average, dvgidt_on_average, dvgidt_off_average,]
            output.append(line)

            print(f'{tss_file} processing is finished.')

    output_df = pd.DataFrame(output)
    output_path = os.path.join(output_folder, output_name + '.csv')
    output_df.to_csv(output_path, index=False, header=False)


def multi_processor_Rdson(input_folders, output_folder, output_name, comment_contains, config_file,
                          t_avg=100e-9, t_trigger=0):
    # Extract all parameter time, energy, qg, dvdt
    output = [
        ['Mark ID', 'Time stamp', 'Comment', 'L (uH)', 'Vd (V)', 'Id (A)', 'inter_pulse_delay (ns)', 'pulse2_width (ns)', "dead_time (ns)",
         "gate_voltage", 'gate_resistor', 'vdd', 'error', 'tdon (ns)', 'tf (ns)', 'tdoff (ns)', 'tr (ns)', 'Eon (uJ)', 'Eoff (uJ)',
        'RDS_1st (mOhm)', 'RDS_2nd (mOhm)', #'Qgon (nC)', 'Qgoff (nC)',
        'Vd_dvdt_on_9010 (V/ns)', 'Vd_dvdt_on_MAX (V/ns)', 'Vd_dvdt_off_1090 (V/ns)', 'Vd_dvdt_off_MAX (V/ns)']
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

            if comment_contains not in comment:
                continue
            tss = TSS_process(tss_folder, tss_file, config_file)

            error = ''
            try:
                energy_output = tss.energy_extraction()
                eon = energy_output[0]
                eoff = energy_output[1]
            except Exception as e:
                eon = 0
                eoff = 0
                error = 'error'

            try:
                time_output = tss.time_extraction()
                tdon = time_output[0]
                tf = time_output[1]
                tdoff = time_output[2]
                tr = time_output[3]
            except Exception as e:
                tdon = 0
                tf = 0
                tdoff = 0
                tr = 0
                error = 'error'

            try:
                rdson_output = tss.rdson_extraction(t_trigger=t_trigger, t_avg=t_avg,
                                                 t_interpulse=inter_pulse_delay*1e-9,
                                                 t_dead=dead_time*1e-9,
                                                 t_2nd=pulse2_width*1e-9)
                rdson1 = rdson_output[0]
                rdson2 = rdson_output[1]
            except Exception as e:
                rdson1 = 0
                rdson2 = 0
                error = 'error'
            """try:
                qg_output = tss.gate_charge_extraction()
                qgoff = qg_output[0]
                qgon = qg_output[1]
            except Exception as e:
                qgoff = 0
                qgon = 0
                error = 'error'"""

            try:
                dvdt_output = tss.dvdt_extraction()
                dvdt_on_average = dvdt_output[0]
                dvdt_on_peak = dvdt_output[1]
                dvdt_off_average = dvdt_output[2]
                dvdt_off_peak = dvdt_output[3]
            except Exception as e:
                dvdt_on_average, dvdt_on_peak, dvdt_off_average, dvdt_off_peak = 0, 0, 0, 0
                error = 'error'

            line = [chip_id, time_stamp, comment, induc, vdp, idp, inter_pulse_delay,
                    pulse2_width, dead_time, gate_voltage, gate_resistor, vdd, error,
                    tdon, tf, tdoff, tr, eon, eoff, rdson1, rdson2, #qgon, qgoff,
                    dvdt_on_average, dvdt_on_peak, dvdt_off_average, dvdt_off_peak]
            output.append(line)

            print(f'{tss_file} processing is finished.')

    output_df = pd.DataFrame(output)
    output_path = os.path.join(output_folder, output_name + '.csv')
    output_df.to_csv(output_path, index=False, header=False)
