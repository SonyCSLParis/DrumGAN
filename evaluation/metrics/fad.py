import torch
import numpy as np
import torch.nn.functional as F
import subprocess
import csv

from utils.utils import GPU_is_available
from utils.utils import list_files_abs_path, mkdir_in_path, checkexists_mkdir
from tqdm import trange, tqdm

from datetime import datetime


def test(parser, visualisation=None):
    args = parser.parse_args()

    if GPU_is_available:
        device = 'cuda'
    else:
        device = 'cpu'


    true_files = list_files_abs_path(args.true_path, 'wav')
    fake_files = list_files_abs_path(args.fake_path, 'wav')
    output_path = args.dir 
    output_path = mkdir_in_path(output_path, "evaluation_metrics")
    output_path = mkdir_in_path(output_path, "fad")

    ###################### COMPUTE FAD ON REAL DATA ######################
    print("Computing FAD on true data...\nYou can skip this with ctrl+c")
    try:

        real_paths_csv = f"{output_path}/real_audio.cvs"
        with open(real_paths_csv, "w") as f:
            for file_path in true_files:
                f.write(file_path + '\n')
        fake_paths_csv = f"{output_path}/fake_audio.cvs"
        with open(fake_paths_csv, "w") as f:
            for file_path in fake_files:
                f.write(file_path + '\n')

        fad = float(subprocess.check_output(["sh",
                            "shell_scripts/run_fad_eval.sh",
                            "--real-path="+real_paths_csv,
                            "--fake-path="+fake_paths_csv,
                            "--model-root-path="+output_path]).decode()[-10:-1])
        with open(f"{output_path}/fad_{len(true_files)}_{datetime.now().strftime('%y_%m_%d')}.txt", "w") as f:
            f.write(str(fad))
            f.close()

        print("FAD_50k={0:.4f}".format(fad))
    except KeyboardInterrupt as k:
        print("Skipping true data FAD")
