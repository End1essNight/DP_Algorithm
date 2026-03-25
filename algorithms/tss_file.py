"""
A class to process TSS files from the Tek MSO54 scope.
"""
import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from boto3 import resource as boto_resource
from .wfm import WFM
from .tss_corrections import get_tss_corrections


def get_zip_contents(zip_object):
    """Returns a list of files in the zip archive.
    zip_object can be a path or a ByteStream object"""
    with ZipFile(zip_object, 'r') as myzip:
        return myzip.namelist()


def unzip_from_archive(zip_object, file):
    """Returns the decompressed file from the zip_object.
    zip_object can be a path or a ByteStream object"""
    with ZipFile(zip_object, 'r') as myzip:
        return myzip.read(file)


def get_local_bytestream(folder, file):
    """Returns the contents of the file as a file-like object"""
    path = Path(folder, file)
    bytes_obj = path.read_bytes()
    return BytesIO(bytes_obj)


def get_s3_bytestream(bucket_name, key):
    """Reads the tss from s3 and return contents as a file-lik object"""
    bucket = boto_resource('s3').__getattribute__('Bucket')(bucket_name)
    obj = bucket.Object(key)
    return BytesIO(obj.get()['Body'].read())


class TSSFile:
    """A class to extract info from a Tektronix scope TSS file"""

    def __init__(self, folder, file, source='local'):
        """Initialise the TSSFile object"""
        self.folder = folder
        self.file = file
        self.source = source
        self.bytestream = self.get_bytestream()
        self.contents = get_zip_contents(self.bytestream)
        self.wfm_files = [x for x in self.contents if x.endswith('.wfm')]
        self.corrections = get_tss_corrections(self.folder, self.file)
        self.channel_labels = self._labels_from_set_file()
        self.waveforms = self.process_wfm_files()
        self.tzero = 0.0
        # Get the time step size from the first channel
        first_channel = tuple(self.channel_labels.keys())[0]
        self.tstep = self.waveforms[first_channel].tstep

    @property
    def nframes(self):
        """Returns the number of frames in the waveforms"""
        first = list(self.waveforms.keys())[0]
        return self.waveforms[first].frame_count

    @property
    def labels(self):
        """Returns a list of labels for available waveforms"""
        return list(self.channel_labels.values())

    def channel_for_label(self, label):
        """Returns the channel for the label"""
        for channel, name in self.channel_labels.items():
            if name == label:
                return channel

    def wfm_for_label(self, label):
        """Returns the WFM object for the label"""
        channel = self.channel_for_label(label)
        if channel is not None:
            return self.waveforms[channel]

    def get_bytestream(self):
        """Returns a filelike object from either a local file or s3"""
        if self.source == 's3':
            return get_s3_bytestream(self.folder, self.file)
        else:
            return get_local_bytestream(self.folder, self.file)

    def process_wfm_files(self):
        """Extract all the WFM objects from the tss file"""
        waveforms = {}
        for wfm_file in self.wfm_files:
            channel = os.path.splitext(wfm_file)[0]
            wfm_bytes = unzip_from_archive(self.bytestream, wfm_file)
            label = self.channel_labels[channel]
            waveforms[channel] = WFM(self.file,
                                     channel,
                                     label,
                                     wfm_bytes)
        return waveforms

    def add_math_wfm(self, label, values):
        """Adds a math waveform with values from the array values"""
        if len(self.wfm_files) < 1:
            return
        # Use the first waveform file to get timing and metadata
        wfm_file = self.wfm_files[0]
        channel = os.path.splitext(wfm_file)[0]
        wfm_bytes = unzip_from_archive(self.bytestream, wfm_file)
        self.channel_labels[label] = label
        self.waveforms[label] = WFM(self.file,
                                    channel,
                                    label,
                                    wfm_bytes)
        self.waveforms[label].values = values

    def _labels_from_set_file(self):
        """Extracting all the set file gives a zip file with the
        same name. So it needs to be unzipped twice.
        Get labels for channels from lines that look like
          :CH1:LABEL:NAME "Vg"
          :CH2:LABEL:NAME "Is"
          :CH3:LABEL:NAME "Vd"
          :CH4:LABEL:NAME "Vgi"
        """
        set_files = [x for x in self.contents if x.endswith(".set")]
        # Get the channel labels from the set file
        if len(set_files) != 1:
            raise ValueError("No set file in TSS file")
        set_file = set_files[0]
        # Unzip the file into a file-like object
        zipdata = BytesIO(unzip_from_archive(self.bytestream, set_file))
        # Unzip file of the same name and convert to lines of str
        try:
            set_file_lines = (
                unzip_from_archive(zipdata, set_file).decode("utf-8").split("\n")
            )
        except KeyError:
            #set_file = f"{set_file.split('.')[0]}_lrn.set"
            set_file = set_file.replace(".set", "_lrn.set")
            set_file_lines = (
                unzip_from_archive(zipdata, set_file).decode("utf-8").split("\n")
            )
        # Filter out lines that contain channel labels and split them
        # This will give ['', 'CH1]', 'LABEL', 'NAME = "Vg"']
        labels = [
            x.strip().split(":")
            for x in set_file_lines
            if "LABEL:NAME" in x
             # and "_" not in x
        ]
        channel_label = {}
        # Create a dictionary of channels and labels
        # Use lower case channels to match wfm file names
        for words in labels:
            name = words[-1].split('"')[1]
            channel = words[1].lower()
            # Only consider standard channels, i.e. not math & have wfm data
            channels_in_wfm = [x.split(".")[0] for x in self.wfm_files]
            if channel.startswith("ch") and channel in channels_in_wfm:
                # In case the label is not set, use the channel name
                if name == "":
                    name = channel
                # Apply a channel name correction if required
                if name in self.corrections["channels"]:
                    name = self.corrections["channels"][name]
                channel_label[channel] = name
        return channel_label