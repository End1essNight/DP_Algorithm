"""
A helper to extract a dictionary from a tss corrections text file
"""
import os
from pathlib import Path
from boto3 import resource as boto_resource


def get_tss_corrections(bucket_name, tss_key):
    """Gets the tss corrections file from the bucket.
    tss_key is the full S3 key of the tss file.
    Gets contents of a file called tss_corrections.txt from
    the folder containing the tss file.
    """
    path = Path(tss_key)
    parent = path.parent
    key = os.path.join(parent, "tss_corrections.txt")
    text = get_s3_text(bucket_name, key)
    return parse_tss_corrections_file(text)


def get_s3_text(bucket_name, key):
    """Reads a text file from s3 and returns a string.
    Returns an empty string if the file does not exist."""
    bucket = boto_resource("s3").__getattribute__("Bucket")(bucket_name)
    try:
        obj = bucket.Object(key)
        return obj.get()["Body"].read().decode("utf-8")
    except Exception:
        return ""


def parse_tss_corrections_file(text):
    """Parses the tss_corrections file
    File is of the format:
        channel:BadName1,NewName1
        channel:BadName2,NewName2
        comment:New comment
    """
    lines = text.split("\n")
    corrections = {"channels": {}, "comment": None}
    for line in lines:
        if line.startswith("channel"):
            names = line.split(":")[-1]
            old, new = [x.strip() for x in names.split(",")]
            corrections["channels"][old] = new
    return corrections
