"""Testing script for the visual grounding pipeline based on segmentation and CLIP."""

import argparse

from torch.utils.data import random_split

from modules.pipelines.clipseg import ClipSeg
from modules.pipelines.clipssd import ClipSSD
from modules.pipelines.detrclip import DetrClip
from modules.pipelines.mdetr import MDETRvg
from modules.refcocog import RefCOCOg
from modules.utilities import visual_grounding_test, get_best_device
from modules.pipelines.yoloclip import YoloClip

supported_pipelines = ["yoloclip", "clipseg", "detrclip", "clipssd", "mdetr"]


def main(args):
    if args.pipeline not in supported_pipelines:
        raise ValueError(f"Pipeline `{args.pipeline}` not supported. Supported pipelines are: {supported_pipelines}.")

    device = get_best_device()

    dataset = RefCOCOg(ds_path=args.datapath)
    test_ds = RefCOCOg(ds_path=args.datapath, split='test')

    if args.red_dataset is not None:
        print(f"[INFO] Reducing dataset to {args.reduce_dataset * 100}% of its original size.")
        keep = args.reduce_dataset
        dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
        test_ds, _ = random_split(test_ds, [int(keep * len(test_ds)), len(test_ds) - int(keep * len(test_ds))])

    print(f"[INFO] Dataset Size: {len(dataset)}")
    print(f"[INFO] test split:   {len(test_ds)}")

    if args.clip_version is None:
        args.clip_version = "RN50"
        print(f"[INFO] No CLIP version specified. Using {args.clip_version}.")
    else:
        print(f"[INFO] Using CLIP version: {args.clip_version}")

    if args.pipeline == "yoloclip":

        if args.yolo_version is None:
            args.yolo_version = "yolov8x"
            print(f"[INFO] No YOLO version specified. Using {args.yolo_version}")

        pipeline = YoloClip(dataset.categories,
                            clip_ver=args.clip_version,
                            yolo_ver=args.yolo_version,
                            device=device)

    if args.pipeline == "clipseg":

        if args.seg_method is None:
            args.seg_method = "w"
            print(f"[INFO] No segmentation method specified. Using Watershed.")
        if args.n_segments is None:
            args.n_segments = (4, 8, 16, 32)
            print(f"[INFO] No number of segments specified. Using {args.n_segments}.")
        if args.threshold is None:
            args.threshold = 0.75
            print(f"[INFO] No threshold specified. Using {args.threshold}.")

        pipeline = ClipSeg(dataset.categories,
                           clip_ver=args.clip_version,
                           method=args.seg_method,
                           n_segments=args.n_segments,
                           q=args.threshold,
                           quiet=True,
                           device=device)

    if args.pipeline == "detrclip":
        pipeline = DetrClip(dataset.categories,
                            clip_ver=args.clip_version,
                            device=device)

    if args.pipeline == "clipssd":

        if args.confidence_t is None:
            raise ValueError(f"Pipeline `{args.pipeline}` need the following arguments:"
                             f"`confidence_t`.")

        pipeline = ClipSSD(dataset.categories,
                           clip_ver=args.clip_version,
                           confidence_t=args.confidence_t,
                           device=device)

    if args.pipeline == "mdetr":
        pipeline = MDETRvg(dataset.categories,
                           clip_ver=args.clip_version,
                           device=device)

    print(f"[INFO] Starting test\n")

    visual_grounding_test(pipeline, test_ds, logging=args.logging)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the visual grounding pipeline.',
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50))

    parser.add_argument('-p', '--pipeline', type=str,
                        help='Pipeline to test (yoloclip or clipseg).')
    parser.add_argument('-dp', '--datapath', type=str, default="dataset/refcocog",
                        help='path to the dataset.')
    parser.add_argument('-lg', '--logging', action='store_true',
                        help='Whether to log the results or not.')
    parser.add_argument('-rd', '--red_dataset', type=float, default=None,
                        help='Whether to use a reduced version of the dataset or not')
    parser.add_argument('-cv', '--clip_version', type=str,
                        help='CLIP version to use (RN50, RN101, ViT-L/14)')
    parser.add_argument('-yv', '--yolo_version', type=str,
                        help='Yolo version to use (yolov5s, yolov8x). [only for yoloclip]')
    parser.add_argument('-sm', '--seg_method', type=str,
                        help='Method to use for segmentation (`s`for SLIC or `w` for Watershed) [only for segclip].')
    parser.add_argument('-ns', '--n_segments', type=list,
                        help='Number of segments to use for segmentation [only for segclip].')
    parser.add_argument('-ts', '--threshold', type=float,
                        help='Threshold for filtering CLIP heatmap [only for segclip].')
    parser.add_argument('-ds', '--downsampling', type=int,
                        help='Heatmap downsampling factor [only for clipseg].')
    parser.add_argument('-ct', '--confidence_t', type=float,
                        help='Confidence t for Single Shot Detection [only for clipssd].')

    args = parser.parse_args()

    main(args)
