import importlib
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        description='Generation testing script', 
                        add_help=False)
    parser.add_argument('generation_test', type=str,
                        help='Name of the generation test (check evaluation/gen_tests) to launch. To get \
                        the arguments specific to an generation test please \
                        use: eval.py evaluation_name -h')
    parser.add_argument('-d', '--dir', help="Path to model's root folder",
                        type=str, dest="dir")
    parser.add_argument('-o', '--out-dir', help='Output directory',
                        type=str, dest="outdir", default="")
    parser.add_argument('-m', '--midi', help='Path to midi file',
                        type=str, dest="midi", 
                        default="./test_midi_files/midi_furelisa_adapted.mid")
    parser.add_argument('-N', '--n-gen', help='Path to midi file',
                        type=int, dest="n_gen", 
                        default=10)
    parser.add_argument('--batch-size', help='Path to midi file',
                        type=int, dest="batch_size", 
                        default=50)
    
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    args, unknown = parser.parse_known_args()

    module = importlib.import_module("evaluation.gen_tests." + args.generation_test)
    print("Running " + args.generation_test)

    parser.add_argument('-h', '--help', action='help')
    out = module.generate(parser)

    if out is not None and not out:
        print("...FAIL")

    else:
        print("...OK")
