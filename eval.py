import importlib
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script', add_help=False)
    parser.add_argument('evaluation_name', type=str,
                        help='Name of the evaluation method to launch. To get \
                        the arguments specific to an evaluation method please \
                        use: eval.py evaluation_name -h')
    parser.add_argument('--no_vis', help='Print more data',
                        action='store_true')
    parser.add_argument('--np_vis', help=' Replace visdom by a numpy based \
                        visualizer (SLURM)',
                        action='store_true')
    parser.add_argument('-m', '--module', help="Module to evaluate, available\
                        modules: PGAN, PPGAN, DCGAN, StyleGAN",
                        type=str, dest="module")
    parser.add_argument('-n', '--name', help="Model's name",
                        type=str, dest="name")
    parser.add_argument('-d', '--dir', help='Output directory',
                        type=str, dest="dir", default="output_networks")
    parser.add_argument('-t', '--test-list', help='Test list: one of [rand_gen, ]',
                        type=int, dest="test_list", nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    
    parser.add_argument('-i', '--iter', help='Iteration to evaluate',
                        type=int, dest="iteration")
    parser.add_argument('-s', '--scale', help='Scale to evaluate',
                        type=int, dest="scale")
    parser.add_argument('-c', '--config', help='Training configuration',
                        type=str, dest="config")
    parser.add_argument('-v', '--partitionValue', help="Partition's value",
                        type=str, dest="partition_value")
    parser.add_argument("-A", "--statsFile", dest="statsFile",
                        type=str, help="Path to the statistics file")
    
    parser.add_argument("-D", "--dataset", dest="data_path",
                        type=str, help="Path to the dataset folder")
    parser.add_argument("-a", "--att_name", dest="att_name",
                        type=str, help="Path to the dataset folder")

    parser.add_argument("--inception-model", dest="imodel",
                        type=str, help="Path to the inception model to use")
    
    parser.add_argument("--sanity-check", dest="scheck", action='store_true',
                        help="For find_z_given_audio, use generated as target")
    
    parser.add_argument("--is", dest="compute_is", action='store_true',
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("--fad", dest="compute_fad", action='store_true',
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("--mmd", dest="compute_mmd", action='store_true',
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("--bounds", dest="compute_bounds", action='store_true',
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("--pmmd", dest="compute_pmmd", action='store_true',
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("--midi", dest="midi_path", type=str,
                        help="For find_z_given_audio, use generated as target")
    
    parser.add_argument("--true_path", dest="true_path", type=str,
                        help="For find_z_given_audio, use generated as target")
    parser.add_argument("--fake_path", dest="fake_path", type=str,
                        help="For find_z_given_audio, use generated as target")

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    args, unknown = parser.parse_known_args()

    vis_module = None
    # if args.np_vis:
    #     vis_module = importlib.import_module("visualization.np_visualizer")
    # elif args.no_vis:
    #     print("Visualization disabled")
    # else:
    #     vis_module = importlib.import_module("models.visualization.visualizer")

    module = importlib.import_module("models.eval." + args.evaluation_name)
    print("Running " + args.evaluation_name)

    parser.add_argument('-h', '--help', action='help')
    out = module.test(parser, visualisation=vis_module)

    if out is not None and not out:
        print("...FAIL")

    else:
        print("...OK")
