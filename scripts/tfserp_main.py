import csv
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tfsenc_config import parse_arguments, setup_environ, write_config
from tfsenc_encoding import build_Y
from tfsenc_load_signal import load_electrode_data
from tfsenc_main import process_electrodes, return_stitch_index
from tfsenc_read_datum import read_datum
from utils import main_timer


def write_erp_results(args, results, filename):
    """Write output into csv files

    Args:
        args (namespace): commandline arguments
        results: correlation results
        filename: usually electrode name plus 'prod' or 'comp'

    Returns:
        None
    """
    filename = os.path.join(args.output_dir, filename)
    df = pd.DataFrame([results])
    df.to_csv(filename, header=False, index=False)
    return


def single_electrode_erp(electrode, args, datum, stitch_index):
    """Doing erp for one electrode

    Args:
        electrode (tuple): ((sid, elec_id), elec_name)
        args (namespace): commandline arguments
        datum (df): datum of words
        stitch_index (list): stitch_index

    Returns:
        tuple in the format (sid, electrode name, production len, comprehension len)
    """
    # Get electrode info
    (sid, elec_id), elec_name = electrode

    if elec_name is None:
        print(f"Electrode ID {elec_id} does not exist")
        return (args.sid, None, 0, 0)

    # Load signal Data
    elec_signal, missing_convos = load_electrode_data(args, elec_id, stitch_index, True)
    if len(missing_convos) > 0:  # modify datum based on missing signal
        elec_datum = datum.loc[
            ~datum["conversation_name"].isin(missing_convos)
        ]  # filter missing convos
    else:
        elec_datum = datum

    if len(elec_datum) == 0:  # datum has no words, meaning no signal
        print(f"{args.sid} {elec_name} No Signal")
        return (args.sid, elec_name, 0, 0)

    # ERP
    Y = build_Y(
        elec_signal.reshape(-1, 1),
        elec_datum.adjusted_onset.values,
        np.array(args.lags),
        args.window_size,
    )
    comp_Y = Y[elec_datum.speaker != "Speaker1", :]
    prod_Y = Y[elec_datum.speaker == "Speaker1", :]
    comp_erp = comp_Y.mean(axis=0)
    prod_erp = prod_Y.mean(axis=0)

    elec_name = str(sid) + "_" + elec_name
    print(f"{args.sid} {elec_name} ({elec_id}), Comp: {len(comp_Y)} Prod: {len(prod_Y)}")

    write_erp_results(args, comp_erp, f"{elec_name}_comp.csv")
    write_erp_results(args, prod_erp, f"{elec_name}_prod.csv")

    return


def electrodes_erp(args, electrode_info, datum, stitch_index, parallel=True):
    """Doing erp for all electrodes

    Args:
        args (namespace): commandline arguments
        electrode_info (dict): dictionary of electrodes
        datum (df): datum of words
        stitch_index (list): stitch_index
    """
    if parallel:
        print("Running all electrodes in parallel")
        with Pool(4) as p:
            p.map(
                partial(
                    single_electrode_erp,
                    args=args,
                    datum=datum,
                    stitch_index=stitch_index,
                ),
                electrode_info.items(),
            )
    else:
        for electrode in electrode_info.items():
            single_electrode_erp(electrode, args, datum, stitch_index)


@main_timer
def main():

    # Read command line arguments
    args, yml_args = parse_arguments()

    # Setup paths to data
    args = setup_environ(args)

    # Saving configuration to output directory
    write_config(args, yml_args)

    # Locate and read datum
    stitch_index = return_stitch_index(args)
    datum = read_datum(args, stitch_index)
    datum.drop(columns=["embeddings"], inplace=True)

    electrode_info = process_electrodes(args)
    electrodes_erp(args, electrode_info, datum, stitch_index, False)

    return


if __name__ == "__main__":
    main()
