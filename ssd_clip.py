# Import necessary packages and set correct device
import os

import numpy as np
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from modules.clipssd import ClipSSD
from modules.refcocog import RefCOCOg, RefCOCOgSample
from modules.utilities import visual_grounding_test


if torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA GPU
    print("[INFO] Using cuda.")
elif torch.has_mps:
    device = torch.device("mps")  # Apple Silicon GPU
    print("[INFO] Using MPS.")
else:
    device = torch.device("cpu")
    print("[INFO] No GPU found, using CPU instead.")






# Import RefCOCOg dataset and its train/val/test splits

print("\nImporting Dataset\n")

data_path = "/media/dmmp/vid+backup/Data/refcocog"
# data_path = "dataset/refcocog"

dataset = RefCOCOg(ds_path=data_path)

# train_ds = RefCOCOg(ds_path=data_path, split='train')
# val_ds = RefCOCOg(ds_path=data_path, split='val')
test_ds = RefCOCOg(ds_path=data_path, split='test')

# keep only a toy portion of each split
keep = 0.1
# red_dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
# red_train_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
# red_val_ds, _ = random_split(val_ds, [int(keep * len(val_ds)), len(val_ds) - int(keep * len(val_ds))])
red_test_ds, _ = random_split(test_ds, [int(keep * len(test_ds)), len(test_ds) - int(keep * len(test_ds))])

# print(f"[INFO] Dataset Size: {len(dataset)}")
# print(f"[INFO] train split:  {len(train_ds)}")
# print(f"[INFO] val split:    {len(val_ds)}")
print(f"[INFO] test split:   {len(red_test_ds)}")

# get path urls
# bboxes = dict()
#
# uris = [os.path.normpath( os.path.join(red_test_ds.dataset.ds_path,"images" , "_".join(path['file_name'].split('_')[0:3]) + ".jpg")) for path in red_test_ds.dataset.refs]
#
# for i in tqdm(range(0, len(uris), 100)):
#     batch = uris[i:i + 100]
#     inputs = [utils.prepare_input(uri) for uri in batch]
#     tensor = utils.prepare_tensor(inputs)
#
#     with torch.no_grad():
#         detections_batch = ssd_model(tensor)
#
#     results_per_input = utils.decode_results(detections_batch)
#     best_results_per_input = [utils.pick_best(results, 0.50) for results in results_per_input]
#     for j in range(len(batch)):
#         bbox, _, _ = best_results_per_input[j]
#         bboxes[batch[j]] = bbox
#
#     pass
# Initialize ClipSeg pipeline





# Test ClipSeg on a random sample

idx = np.random.randint(0, len(red_test_ds))
sample = RefCOCOgSample(**red_test_ds[idx])
print("Loading solver")
solver = ClipSSD(dataset.categories, confidence_t=0.01, device=device)
print("Testing on 1 image")
solver(sample, sample.sentences[0], show=True, timeit=False)


# Execute testing on the test dataset

visual_grounding_test(solver, red_test_ds)