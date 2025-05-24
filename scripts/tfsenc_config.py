import argparse
import getpass
import os

import numpy as np
import torch
import yaml
from himalaya.backend import set_backend
from utils import get_git_hash
from utils import get_dir

ELEC_SIGNAL_PREPROCESS_MAP = {
    "podcast": dict.fromkeys(
        [
            "661",
            "662",
            "717",
            "723",
            "737",
            "741",
            "742",
            "743",
            "763",
            "798",
            "777",
        ],
        "preprocessed_all",
    ),
    "tfs": dict.fromkeys(["625", "676"], "preprocessed")
    | dict.fromkeys(["7170"], "preprocessed_v2")
    | dict.fromkeys(["798"], "preprocessed_allElec"),
}

ELEC_SIGNAL_FOLDER_MAP = {
    "podcast": "/projects/HASSON/247/data/podcast-data",
    "tfs": "/projects/HASSON/247/data/conversations-car",
}


def clean_lm_model_name(item):
    """Remove unnecessary parts from the language model name.

    Args:
        item (str/list): full model name from HF Hub

    Returns:
        (str/list): pretty model name

    Example:
        clean_lm_model_name(EleutherAI/gpt-neo-1.3B) == 'gpt-neo-1.3B'
    """
    if isinstance(item, str):
        return item.split("/")[-1]

    if isinstance(item, list):
        return [clean_lm_model_name(i) for i in item]

    print("Invalid input. Please check.")


def parse_arguments():
    """Read arguments from yaml config file

    Returns:
        namespace: all arguments from yaml config file
    """
    # parse yaml config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", nargs="*", type=str, default="[config.yml]")
    args = parser.parse_args()

    all_yml_args = {}
    for config_file in args.config_file:
        with open(f"{get_dir('configs')}/{config_file}", "r") as file:
            yml_args = yaml.safe_load(file)
            all_yml_args = all_yml_args | yml_args

    # get username
    user_id = getpass.getuser()
    all_yml_args["user_id"] = user_id
    # all_yml_args["git_hash"] = get_git_hash()

    # clean up args
    args = argparse.Namespace(**all_yml_args)
    try:  # eval lists
        args.elecs = eval(args.elecs)
        args.conv_ids = eval(args.conv_ids)
        args.lags = eval(args.lags)
    except:
        print("List parameter failed to eval")

    if args.emb == "glove50":  # for glove, fix layer and context len
        args.layer_idx = 0 #1
        args.context_length = 1
    elif args.emb == "glove50-nm":  # Not my glove, fix layer and context len
        args.layer_idx = 1
        args.context_length = 1
    else:
        args.emb = clean_lm_model_name(args.emb)

    if hasattr(args, "l0"):
        args.l0 = int(args.l0)

    return args, all_yml_args


def setup_environ(args):
    """
    Update args with project specific directories and other flags

    Args:
        args (namespace): arguments

    Returns:
        args (namespace): arguments plus directory paths
    """

    # input directory paths (pickles)
    DATA_DIR = get_dir("data")
    PICKLE_DIR = os.path.join(DATA_DIR, args.project_id, str(args.sid), "pickles")
    EMB_DIR = os.path.join(PICKLE_DIR, "embeddings", args.emb, "full")
    args.base_df_path = os.path.join(EMB_DIR, "base_df.pkl")
    EMB_DF_DIR = os.path.join(EMB_DIR, f"cnxt_{args.context_length:04d}")
    if hasattr(args, "width"):
        EMB_DF_DIR = os.path.join(EMB_DF_DIR, f"width_{args.width}")
    if hasattr(args, "l0"):
        EMB_DF_DIR = os.path.join(EMB_DF_DIR, f"l0_{args.l0:03d}")

    args.emb_df_path = os.path.join(
        EMB_DF_DIR,
        f"layer_{args.layer_idx:02d}.pkl",
)
    args.electrode_file_path = os.path.join(
        PICKLE_DIR, ("_".join([str(args.sid), "electrode_names.pkl"]))
    )
    if args.sig_elec_file is not None:
        args.sig_elec_file_path = os.path.join(DATA_DIR, args.sig_elec_file)
    args.stitch_file_path = os.path.join(
        PICKLE_DIR, ("_".join([str(args.sid), "full_stitch_index.pkl"]))
    )

    # input directory paths (elec mat files)
    ELEC_SIGNAL_DIR = ELEC_SIGNAL_FOLDER_MAP[args.project_id]
    args.elec_signal_file_path = os.path.join(
        ELEC_SIGNAL_DIR, str(args.sid), "NY*Part*conversation*"
    )
    args.elec_signal_process_flag = ELEC_SIGNAL_PREPROCESS_MAP[args.project_id][
        str(args.sid)
    ]

    # output directory paths
    OUTPUT_DIR = get_dir(os.path.join("results", args.project_id))
    RESULT_PARENT_DIR = f"{args.user_id[0:2]}-{args.project_id}-{args.sid}-{args.emb}-{args.output_dir_name}"
    res_dir_base = f"{args.user_id[0:2]}-{args.window_size}ms-{args.sid}-lay{args.layer_idx}-con{args.context_length}"
    if args.regularization in {"ridge", "lasso"}:
        res_dir_base += f"-reg{args.regularization}"
    elif args.pca_to:
        res_dir_base += f"-pca{args.pca_to}"
    if args.emb_norm:
        res_dir_base += f"-norm{args.emb_norm}"
    RESULT_CHILD_DIR = res_dir_base
    args.output_dir = os.path.join(OUTPUT_DIR, RESULT_PARENT_DIR, RESULT_CHILD_DIR)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        print("set backend to cuda")
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        print("set backend to cpu numpy")
        backend = set_backend("numpy", on_error="warn")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # HACK

    return args


def write_config(args, yml_args):
    """Write configuration to a file

    Args:
        args (namespace): configuration
        yml_args (dict): original yml args
    """
    config_file_path = os.path.join(args.output_dir, "config.yml")
    with open(config_file_path, "w") as outfile:
        yaml.dump(yml_args, outfile, default_flow_style=False, sort_keys=False)
