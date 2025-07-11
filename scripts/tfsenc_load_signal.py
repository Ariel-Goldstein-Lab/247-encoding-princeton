import glob
import os

import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def detrend_signal(mat_signal):  # Detrending
    """Detrends a signal

    Args:
        mat_signal: signal for a specific conversation

    Returns:
        mat_signal: detrended signal
    """

    y = mat_signal
    X = np.arange(len(y)).reshape(-1, 1)
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)

    model = LinearRegression()
    model.fit(Xp, y)
    trend = model.predict(Xp)
    mat_signal = y - trend

    return mat_signal


def create_nan_signal(stitch, convo_id):
    """Returns fake signal for a conversation

    Args:
        stitch: stitch_index
        convo_id: conversation id

    Returns:
        mat_signal: nans of a specific conversation size
    """

    mat_len = stitch[convo_id] - stitch[convo_id - 1]  # mat file length
    mat_signal = np.empty((mat_len, 1))
    mat_signal.fill(np.nan)

    return mat_signal


def load_electrode_data(args, sid, elec_id, stitch, z_score=False):
    """Load and concat signal mat files for a specific electrode

    Args:
        args (namespace): commandline arguments
        elec_id: electrode id
        stitch: stitch_index
        z_score: whether we z-score the signal per conversation

    Returns:
        elec_signal: concatenated signal for a specific electrode
        elec_datum: modified datum based on the electrode signal
    """
    elec_signal_file_path = os.path.join(
        args.elec_signal_dir, str(sid), "NY*Part*conversation*"
    )
    convos = sorted(glob.glob(elec_signal_file_path))

    all_signal = []
    missing_convos = []
    for convo_id, convo in enumerate(convos, start=1):

        if convo_id not in args.conv_ids:  # skip convo
            continue

        file = glob.glob(
            os.path.join(
                convo, args.elec_signal_process_flag, "*_" + str(elec_id) + ".mat"
            )
        )
        assert (
            len(file) < 2
        ), f"More than 1 signal file exists for electrode {elec_id} at {convo}"

        if len(file) == 1:  # conversation mat file exists
            file = file[0]
            mat_signal = loadmat(file)["p1st"]
            mat_signal = mat_signal.reshape(-1, 1)

            if mat_signal is None:
                continue
            if args.detrend_signal:  # detrend conversation signal
                mat_signal = detrend_signal(mat_signal)
            if z_score:  # z-score when doing erp
                mat_signal = stats.zscore(mat_signal)

        elif len(file) == 0:  # conversation mat file does not exist
            missing_convos.append(os.path.basename(convo))  # append missing convo name
            mat_signal = create_nan_signal(stitch, convo_id)

        all_signal.append(mat_signal)  # append conversation signal

    if args.project_id == "tfs":
        elec_signal = np.vstack(all_signal)
    else:
        elec_signal = np.array(all_signal)

    return elec_signal, missing_convos
