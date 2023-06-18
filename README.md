# Deep Learning Lab 2023 - Visual Grounding

### Problem formulation

Visual grounding (also known as phrase grounding) aims to predict the location of a region referred by the language
expression onto an image.

Traditionally, the problem can be formulated both as a:

- **Two-stage problem**: a first detection stage denotes any off-the-shelf object detector for extracting candidate
  regions from the image, and the second visual grounding step serves to rank the candidates and select the top one
  based on the similarities to the query.
- **One-stage problem**: the two steps are unified and as single model is trained to directly predict the bounding box
  of the object referred by the query.

In this scenario, the dataset we've been using is **RefCOCOg**.

### About the repo

In this repository you can explore different ways to address this task, mainly based on the use of CLIP. The different
pipelines can be found under `modules/pipelines/` and they include:

- **YoloClip**, a baseline pipeline using YOLO for object proposals and CLIP for the grounding task;
- **SegClip**, a pipeline which involves image segmentation, CLIP embedding and bounding box proposal;
- **DetrClip**, a pipeline using DETR for object proposals and CLIP for the grounding task;
- **MDETR**, reported for its SOTA results on phrase grounding, yet not using CLIP;
- **ClipSSD**, a pipeline using Single Shot Detector for object proposals and CLIP for the grounding task;

Other than those, under `modules/` you can also find code for:

- a framework involving **Reinforcement Learning** for bounding box regression and CLIP as a feature extractor;
- some experiments on using **Diffusion Models** for bounding box regression;

Moreover:

- `test.py` can be used to test one of the above mentioned pipelines on the test dataset. Please refer to `--help` for
  more information;
- `pipelines_zoo.ipynb` can be used to experiment around with the different pipelines in a Jupyter Notebook;
- lastly, `train.py` provides some code for fine-tuning CLIP on the dataset, using contrastive learning on each
  detection-text pair. Please refer to `--help` for more information.

### Literature, useful links, and more

- [Deconfounded Visual Grounding, Huang et al.](https://ink.library.smu.edu.sg/sis_research/7484/)

Existing grounding methods are affected by an often overlooked bias, as the bounding box prediction tends to be biased
towards some particular regions of the image. A possible solution is implemented as a Referring Expression
Deconfounder (RED) and compared to other SOTA methods.

- [Adapting CLIP For Phrase Localization Without Further Training, Jiahao et al.](http://arxiv.org/abs/2204.03647)

Adapt a pre-trained CLIP model for the task of phrase localization, employing a Region-based Attention Pooling. CLIP is
trained to output an embedding vector for a given image or text phrase, hence the image embedding cannot be directly
applied to phrase localization which requires spatial reasoning. After obtaining these spatial features, for each pixel
location, we compute the inner product between the spatial feature and the text embedding extracted from CLIP to obtain
a score map. Finally, we predict the bounding box that have the largest score according to the extracted map.

- [Visual Grounding with Transformers, Ye et al.](https://ieeexplore.ieee.org/document/9859880)

- [TransVG: End-to-End Visual Grounding with Transformers, Jiajun et al.](https://ieeexplore.ieee.org/document/9710016)

The first transformer-based framework proposal for visual grounding task.

- [Dynamic MDETR: A Dynamic Multimodal Transformer Decoder for Visual Grounding, Fengyuan et al.](https://arxiv.org/abs/2209.13959)

Introduction of a new multimodal transformer architecture for visual grounding, termed as Dynamic MDETR, which is based
on a relatively shallow encoder for cross-modal feature fusion and alignment, and a dynamic decoder for efficient
text-guided visual localization.

- [Visual Grounding via Accumulated Attention, Chaorui et al.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Visual_Grounding_via_CVPR_2018_paper.pdf)

- [Link](https://github.com/openai/CLIP) to the OpenAI CLIP repo
- [Link](https://pytorch.org/hub/ultralytics_yolov5/) to the TorchHub YOLO page
- [Link](https://github.com/halixness/understanding-CLIP) some nice notes on CLIP

- [Learning transferable visual models from natural language supervision, Radford et al. (2021)](http://proceedings.mlr.press/v139/radford21a/radford21a.pdf)

- [Path aggregation network for instance segmentation, Liu et al. (2018)](https://arxiv.org/abs/1803.01534)

- [Modeling context in referring expressions, Yu et al. (2016)](https://arxiv.org/abs/1608.00272)

- [TheShadow29/awesome-grounding](https://github.com/TheShadow29/awesome-grounding) - A curated list of awesome visual
  grounding resources
