import argparse
import yaml
from train_eval.visualizer import Visualizer
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
parser.add_argument("-o", "--output_dir", help="Directory to save results", required=True)
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=True)
parser.add_argument("--num_modes", help="Number of modes to visualize", type=int, default=10)
parser.add_argument("--examples", help="Example to visualize", type=str, default=1)
parser.add_argument("--frames", help="Frames to visualize", type=str, default=None)
parser.add_argument("--tf", help="Prediction horizon in seconds", type=int, default=6)
parser.add_argument("--show_predictions", help="Show predictions", action="store_true", default=True)
parser.add_argument("--counterfactual", help="Include counterfactual", action="store_true")
parser.add_argument("--mask_lane", help="Mask gt lanes", action="store_true")
parser.add_argument("--name", type=str, default='')
args = parser.parse_args()

# Convert comma-separated string to list of ints
if args.examples is not None:
    args.examples = [int(x) for x in args.examples.split(',')]
if args.frames is not None:
    args.frames = [int(x) for x in args.frames.split(',')]

# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'results')):
    os.mkdir(os.path.join(args.output_dir, 'results'))


# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)


# Visualize
vis = Visualizer(cfg, args.data_root, args.data_dir, args.checkpoint, args.examples, args.frames, args.show_predictions,
                 args.tf, args.num_modes, args.counterfactual, args.mask_lane, args.name)
vis.visualize(output_dir=args.output_dir, dataset_type=cfg['dataset'])
