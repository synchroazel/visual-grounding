"""Testing script for the visual grounding pipeline based on segmentation and CLIP."""

import torch
from torch.utils.data import random_split

from modules.yoloclip import YoloClip
from modules.clipseg import ClipSeg
from modules.refcocog import RefCOCOg
from modules.utilities import visual_grounding_test

import argparse

DATA_PATH = "dataset/refcocog"  # path to the dataset


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
    device = get_best_device()

    dataset = RefCOCOg(ds_path=DATA_PATH)

    train_ds = RefCOCOg(ds_path=DATA_PATH, split='train')
    val_ds = RefCOCOg(ds_path=DATA_PATH, split='val')
    test_ds = RefCOCOg(ds_path=DATA_PATH, split='test')

    keep = 0.1
    red_dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
    red_train_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
    red_val_ds, _ = random_split(val_ds, [int(keep * len(val_ds)), len(val_ds) - int(keep * len(val_ds))])
    red_test_ds, _ = random_split(test_ds, [int(keep * len(test_ds)), len(test_ds) - int(keep * len(test_ds))])

    print(f"[INFO] Dataset Size: {len(dataset)}")
    print(f"[INFO] train split:  {len(train_ds)}")
    print(f"[INFO] val split:    {len(val_ds)}")
    print(f"[INFO] test split:   {len(test_ds)}")

    print(f"[INFO] Starting testing\n")

    yoloclip = YoloClip(dataset.categories, yolo_ver="yolov8x",
                        quiet=True, device=device)

    clipslic = ClipSeg(dataset.categories, method="w", n_segments=(4, 8, 16, 32), q=0.75,
                       quiet=True, device=device)

    visual_grounding_test(clipslic, test_ds, logging=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the visual grounding pipeline.')

    parser.add_argument('-p', '--pipeline', type=str,
                        help='Pipeline to test (yoloclip or segclip).')
    parser.add_argument('-d', '--datapath', type=str,
                        help='path to the dataset.')
    parser.add_argument('-v', '--yolo_version', type=str,
                        help='Yolo version to use (yolov5x, yolov8x). [only for yoloclip]')
    parser.add_argument('-s', '--seg_method', type=str,
                        help='Method to use for segmentation (`s`for SLIC or `w` for Watershed) [only for segclip].')
    parser.add_argument('-n', '--n_segments', type=int,
                        help='Number of segments to use for segmentation [only for segclip].')
    parser.add_argument('-q', '--q', type=float,
                        help='Threshold for filtering CLIP heatmap [only for segclip].')
    parser.add_argument('-d', '--test_size', type=int,
                        help='Heatmap downsampling factor [only for segclip].')

    args = parser.parse_args()

    main(args)
