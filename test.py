"""Testing script for the visual grounding pipeline based on segmentation and CLIP."""

import argparse

import torch
from torch.utils.data import random_split

from modules.clipseg import ClipSeg
from modules.refcocog import RefCOCOg
from modules.utilities import visual_grounding_test
from modules.yoloclip import YoloClip

hp_presets = {
    "yoloclip": {'yolo_ver': 'yolov8x'},
    "clipseg": {'method': "w", 'n_segments': (4, 8, 16, 32), 'q': 0.75}
}

supported_pipelines = ["yoloclip", "clipseg"]


def get_best_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # CUDA GPU
        print("[INFO] Using cuda.")
    elif torch.has_mps:
        device = torch.device("mps")  # Apple Silicon GPU
        print("[INFO] Using MPS.")
    else:
        device = torch.device("cpu")
        print("[INFO] No GPU found, using CPU instead.")

    return device


def main(args):
    if args.pipeline not in supported_pipelines:
        raise ValueError(f"Pipeline `{args.pipeline}` not supported. Supported pipelines are: {supported_pipelines}.")

    device = get_best_device()

    dataset = RefCOCOg(ds_path=args.datapath)
    test_ds = RefCOCOg(ds_path=args.datapath, split='test')

    if args.reduce_dataset is not None:
        print(f"[INFO] Reducing dataset to {args.reduce_dataset * 100}% of its original size.")
        keep = args.reduce_dataset
        red_dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
        red_test_ds, _ = random_split(test_ds, [int(keep * len(test_ds)), len(test_ds) - int(keep * len(test_ds))])
        print(f"[INFO] Dataset Size (reduced): {len(red_dataset)}")
        print(f"[INFO] test split (reduced):   {len(red_test_ds)}")

    else:
        print(f"[INFO] Dataset Size: {len(dataset)}")
        print(f"[INFO] test split:   {len(test_ds)}")

    if args.pipeline == "yoloclip":

        if args.use_preset:
            pipeline = YoloClip(dataset.categories, **hp_presets["yoloclip"], quiet=True, device=device)

    if args.pipeline == "clipseg":

        if args.use_preset:
            pipeline = ClipSeg(dataset.categories, **hp_presets["clipseg"], quiet=True, device=device)

    print(f"[INFO] Starting testing\n")

    visual_grounding_test(pipeline, test_ds, logging=args.logging)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the visual grounding pipeline.',
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50))

    parser.add_argument('-p', '--pipeline', type=str,
                        help='Pipeline to test (yoloclip or segclip).')
    parser.add_argument('-dp', '--datapath', type=str,
                        help='path to the dataset.')
    parser.add_argument('-lg', '--logging', action='store_true',
                        help='Whether to log the results or not.')
    parser.add_argument('-rd', '--red_dataset', type=float, default=None,
                        help='Whether to use a reduced version of the dataset or not')
    parser.add_argument('-up', '--use_preset', action='store_true',
                        help='Whether to use a preset of hyperparameters for the chosen pipeline or not.')
    parser.add_argument('-yv', '--yolo_version', type=str,
                        help='Yolo version to use (yolov5x, yolov8x). [only for yoloclip]')
    parser.add_argument('-sm', '--seg_method', type=str,
                        help='Method to use for segmentation (`s`for SLIC or `w` for Watershed) [only for segclip].')
    parser.add_argument('-ns', '--n_segments', type=int,
                        help='Number of segments to use for segmentation [only for segclip].')
    parser.add_argument('-ts', '--threshold', type=float,
                        help='Threshold for filtering CLIP heatmap [only for segclip].')
    parser.add_argument('-ds', '--downsampling', type=int,
                        help='Heatmap downsampling factor [only for clipseg].')

    args = parser.parse_args()

    main(args)
