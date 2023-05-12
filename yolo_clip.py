import numpy as np
import torch
from tqdm import tqdm

from modules.refcocog import RefCOCOg, RefCOCOgSample
from modules.yoloclip import YoloClip

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("[INFO] GPU found, using GPU.")
else:
    device = torch.device("cpu")
    print("[INFO] No GPU found, using CPU instead.")

# data_path = "/media/dmmp/vid+backup/Data/refcocog"
data_path = "../dataset/refcocog"

dataset = RefCOCOg(ds_path=data_path)

train_ds = RefCOCOg(ds_path=data_path, split='train')
val_ds = RefCOCOg(ds_path=data_path, split='val')
test_ds = RefCOCOg(ds_path=data_path, split='test')

print(f"Dataset Size: {len(dataset)}\n")
print(f"Train size: {len(train_ds)}")
print(f"Val size:   {len(val_ds)}")
print(f"Test size:  {len(test_ds)}")

yoloclip = YoloClip(device="cpu", categories=dataset.categories)

idx = np.random.randint(0, len(dataset))

sample = RefCOCOgSample(**dataset[idx])

for sentence in sample.sentences:
    yoloclip(sample, sentence, show=True)


def visual_grounding_test(vg_pipeline, dataset, track="IoU"):
    scores = list()

    pbar = tqdm(dataset)

    for sample in pbar:

        sample = RefCOCOgSample(**sample)

        for sentence in sample.sentences:
            sc = vg_pipeline(sample, sentence, show=False)

            scores.append(sc)

            avg_score = np.mean([score[track] for score in scores])

            pbar.set_description(f"Average {track}: {avg_score:.3f}")

    for metric in scores[0].keys():
        avg_metric = np.mean([score[metric] for score in scores])

        print("Avg. {}: {:.3f}".format(metric, avg_metric))


yoloclip.quiet = True

visual_grounding_test(yoloclip, test_ds)
