import importlib
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        description='Testing script', add_help=False)
    parser.add_argument('evaluation_metric', type=str,
                        help='Name of the evaluation method to launch. To get \
                        the arguments specific to an evaluation method please \
                        use: eval.py evaluation_name -h')
    parser.add_argument('-d', '--dir', help='Output directory',
                        type=str, dest="dir")
    parser.add_argument('-i', '--iter', help='Iteration to evaluate',
                        type=int, dest="iteration")
    parser.add_argument('-s', '--scale', help='Scale to evaluate',
                        type=int, dest="scale")
    parser.add_argument("--inception-model", dest="inception_model",
                        type=str, help="Path to the inception model to use")
    parser.add_argument("--true", dest="true_path", type=str,
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("--fake", dest="fake_path", type=str,
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("-o", "--outdir", dest="outdir", type=str, default="",
                        help="ooutput directory")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=50,
                        help="ooutput directory")

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    args, unknown = parser.parse_known_args()

    module = importlib.import_module(
        "evaluation.metrics." + args.evaluation_metric)
    print("Running " + args.evaluation_metric)

    parser.add_argument('-h', '--help', action='help')
    out = module.test(parser)

    if out is not None and not out:
        print("...FAIL")
    else:
        print("...OK")
