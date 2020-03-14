import subprocess

from utils.utils import GPU_is_available
from utils.utils import list_files_abs_path, mkdir_in_path

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
    real_paths_csv = f"{output_path}/real_audio.cvs"
    with open(real_paths_csv, "w") as f:
        for file_path in true_files:
            f.write(file_path + '\n')
    fake_paths_csv = f"{output_path}/fake_audio.cvs"
    with open(fake_paths_csv, "w") as f:
        for file_path in fake_files:
            f.write(file_path + '\n')

    fad = float(subprocess.check_output(["sh",
                        "shell_scripts/fad.sh",
                        "--real="+real_paths_csv,
                        "--fake="+fake_paths_csv,
                        "--output="+output_path]).decode()[-10:-1])
    with open(f"{output_path}/fad_{len(true_files)}_{datetime.now().strftime('%y_%m_%d')}.txt", "w") as f:
        f.write(str(fad))
        f.close()

    print("FAD={0:.4f}".format(fad))
