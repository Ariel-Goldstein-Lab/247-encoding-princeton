import os
import pickle
import subprocess
from datetime import datetime


def get_git_hash() -> str:
    """Get git hash as string"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main_timer(func):
    def function_wrapper():
        start_time = datetime.now()
        print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return function_wrapper


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        filename (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        Dictionary: pickle contents returned as dataframe
    """
    print(f"Loading {file}")
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    return datum

def get_dir(path_ending):
    """Get the path a directory. Used to allow running through Makefile or through script directly (e.g. for debugging)"""
    current_dir = os.getcwd()
    # Check if we're in the main directory (with data/ as a direct subdirectory)
    if os.path.isdir(os.path.join(current_dir, path_ending)):
        return os.path.join(current_dir, path_ending)
    # Check if we're in a script/ subdirectory (need to go one level up)
    elif os.path.basename(current_dir) == "scripts" and os.path.isdir(os.path.join(os.path.dirname(current_dir), path_ending)):
        return os.path.join(os.path.dirname(current_dir), path_ending)
    # If neither condition is met, raise an error
    else:
        raise FileNotFoundError(f"Could not locate the data directory ({path_ending})")