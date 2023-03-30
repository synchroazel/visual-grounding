# Visual Grounding

Visual Grounding refers to locating the most relevant region in an image based on a natural language query.

### Metrics to evaluate

Some good metrics to consider include

- **localization accuracy** (how accurately the system can localize an object in the image?)
    - *Intersection over Union (IoU)* is used to measure localization accuracy
- **grounding accuracy** (how accurately it can ground the localized object to a language description?)
    - *Recall* is used to measure grounding accuracy
- **semantic similarity** (how similar are the predicted bounding boxes and the ground-truth descriptions?)
    - *Cosine similarity* and *Euclidean distance* are used to measure semantic similarity

# TODO