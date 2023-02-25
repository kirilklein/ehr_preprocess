import pandas as pd
import os, sys


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def get_csv_files_from_data_dir(data_dir, exclude_files=[]):   
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv.gz') and not f in exclude_files]
    return files

def pandas_get_columns(path):
    if path.endswith('.gz'):
        compression = 'gzip'
    return pd.read_csv(path, nrows=1, compression=compression).columns

drop_columns_dic = {
    'CHARTEVENTS.csv.gz': ['RESULTSTATUS', 'STOPPED','WARNING', 'STORETIME', 'CGID', 'VALUENUM', 'ROW_ID'],
    'DATETIMEEVENTS.csv.gz': ['ROW_ID', 'VALUEUOM', 'WARNING', 'RESULTSTATUS', 'STOPPED',]
}
def prepend(input, prepend_token, slice_start=None, slice_end=None):
    return prepend_token + slice_wrapper(input, slice_start, slice_end)

def slice_wrapper(input, slice_start=None, slice_end=None):
    return input[slice_start:slice_end]

def timing_function(function):
    """
    A decorator that prints the time a function takes to execute.
    """
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = function(*args, **kwargs)
        t2 = time.time()
        print(f'{function.__qualname__!r}: {(t2 - t1)/60:.1f} mins')
        return result
    return wrapper