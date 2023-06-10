import math
import os
import pickle
import random
import shutil

import PIL.Image
from collections import namedtuple

import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm

from refcocog import RefCOCOg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from clip import clip
from utilities import cosine_similarity

# strucure taken from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://arxiv.org/pdf/2208.04511.pdf

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state',))


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)


BBOX_LOW_LIMIT = 0
BBOX_HIGH_LIMIT = 1000


def move_right(bbox, alpha: float) -> tuple:
    Aw = abs(alpha * (bbox[2] - bbox[0]))
    # Ah = alpha * (bbox[3] - bbox[1])

    right = bbox[2] + Aw if bbox[2] + Aw < BBOX_HIGH_LIMIT else BBOX_HIGH_LIMIT

    return [bbox[0] + Aw, bbox[1], right, bbox[3]], True


def move_left(bbox, alpha: float) -> tuple:
    Aw = abs(alpha * (bbox[2] - bbox[0]))
    # Ah = alpha * (bbox[3] - bbox[1])

    left = bbox[0] - Aw if bbox[0] - Aw > BBOX_LOW_LIMIT else BBOX_LOW_LIMIT

    return [left, bbox[1], bbox[2] - Aw, bbox[3]], True


def move_up(bbox, alpha: float) -> tuple:
    # Aw = alpha * (bbox[2] - bbox[0])
    Ah = abs(alpha * (bbox[3] - bbox[1]))

    up = bbox[1] - Ah if bbox[1] - Ah > BBOX_LOW_LIMIT else BBOX_LOW_LIMIT
    return [bbox[0], up, bbox[2], bbox[3] - Ah], True


def move_down(bbox, alpha: float) -> tuple:
    # Aw = abs(alpha * (bbox[2] - bbox[0]))
    Ah = abs(alpha * (bbox[3] - bbox[1]))

    down = bbox[3] + Ah if bbox[3] + Ah < BBOX_HIGH_LIMIT else BBOX_HIGH_LIMIT
    return [bbox[0], bbox[1] + Ah, bbox[2], down], True


def make_bigger(bbox, alpha: float) -> tuple:
    Aw = abs(alpha * (bbox[2] - bbox[0]))
    Ah = abs(alpha * (bbox[3] - bbox[1]))

    left = bbox[0] - Aw if bbox[0] - Aw > BBOX_LOW_LIMIT else BBOX_LOW_LIMIT
    right = bbox[2] + Aw if bbox[2] + Aw < BBOX_HIGH_LIMIT else BBOX_HIGH_LIMIT
    up = bbox[1] - Ah if bbox[1] - Ah > BBOX_LOW_LIMIT else BBOX_LOW_LIMIT
    down = bbox[3] + Ah if bbox[3] + Ah < BBOX_HIGH_LIMIT else BBOX_HIGH_LIMIT

    return [left, up, right, down], True


def make_smaller(bbox, alpha: float) -> tuple:
    Aw = abs(alpha * (bbox[2] - bbox[0]))
    Ah = abs(alpha * (bbox[3] - bbox[1]))

    return [bbox[0] + Aw, bbox[1] + Ah, bbox[2] - Aw, bbox[3] - Ah], True


def make_fatter(bbox, alpha: float) -> tuple:
    # Aw = alpha * (bbox[2] - bbox[0])
    Ah = abs(alpha * (bbox[3] - bbox[1]))

    return [bbox[0], bbox[1] + Ah, bbox[2], bbox[3] - Ah], True


def make_taller(bbox, alpha: float) -> tuple:
    Aw = abs(alpha * (bbox[2] - bbox[0]))
    # Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0] + Aw, bbox[1], bbox[2] - Aw, bbox[3]], True


def stop(bbox, *args):
    return bbox, False


def get_image_center_bb(image_w, image_h):
    w_quarter = image_w / 4
    h_quarter = image_h / 4

    return [w_quarter, h_quarter, image_w - w_quarter, image_h - h_quarter]


def grounding_accuracy(img_encoding, category_encodings: dict, true_category: str):
    all_c_sims = dict()

    for category_id in category_encodings.keys():
        cur_categ_enc = category_encodings[category_id].float()

        all_c_sims[category_id] = cosine_similarity(img_encoding, cur_categ_enc)

    pred_category = max(all_c_sims, key=all_c_sims.get)

    return 1 if pred_category == true_category else 0


class RL_Clip(nn.Module):
    def __init__(self, actions=("left", 'right', 'top', 'down', 'bigger', 'smaller', 'fatter', 'taller', 'trigger'),
                 clip_ver="RN101", device='cuda',
                 maximum_past_actions_memory=10, random_factor=0.9, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.possible_actions = actions
        self.actions_code = {"left": 0, 'right': 1, 'top': 2, 'down': 3, 'bigger': 4, 'smaller': 5, 'fatter': 6,
                             'taller': 7, 'trigger': 8}
        self.actions_dummies = {"left": (1, 0, 0, 0, 0, 0, 0, 0, 0),
                                'right': (0, 1, 0, 0, 0, 0, 0, 0, 0),
                                'top': (0, 0, 1, 0, 0, 0, 0, 0, 0),
                                'down': (0, 0, 0, 1, 0, 0, 0, 0, 0),
                                'bigger': (0, 0, 0, 0, 1, 0, 0, 0, 0),
                                'smaller': (0, 0, 0, 0, 0, 1, 0, 0, 0),
                                'fatter': (0, 0, 0, 0, 0, 0, 1, 0, 0),
                                'taller': (0, 0, 0, 0, 0, 0, 0, 1, 0),
                                'trigger': (0, 0, 0, 0, 0, 0, 0, 0, 1),
                                }
        self.actions = {"left": move_left, 'right': move_right, 'top': move_up, 'down': move_down,
                        'bigger': make_bigger,
                        'smaller': make_smaller, 'fatter': make_fatter, 'taller': make_taller, 'trigger': stop}
        self.clip_model, self.clip_prep = clip.load(clip_ver, device=device)
        self.IoU_treshold = 0.5
        self.trigger_final_reward = 3
        self.maximum_past_actions_memory = maximum_past_actions_memory
        self.past_actions = torch.zeros((1, len(actions) * self.maximum_past_actions_memory)).to(
            device)  # past action tensor, one-hot encoded
        self.lr = 0.001  # learning rate
        self.alpha = 0.3
        self.random_factor = random_factor
        self.buffer_experience_replay = 15_000
        self._memory_update_index = 0

        # start the rewards table
        self.replay = []
        # G_state = G_state + α(target — G_state)

        self.policy_net = DQN(512 * 2 + len(actions) * self.maximum_past_actions_memory, len(actions)).to(device)
        self.target_net = DQN(512 * 2 + len(actions) * self.maximum_past_actions_memory, len(actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        self.steps_done = 0
        self.EPS_END = 0.05
        self.EPS_START = 0.9
        self.EPS_DECAY = 1000

    def choose_action(self, bbox, ground_truth_bbox) -> str:
        next_move = None
        randomN = random.random()
        if randomN < self.random_factor:
            # if random number below random factor, choose random action
            next_move = np.random.choice(self.possible_actions)
        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            good_actions = []
            for action in self.possible_actions:
                new_bbox, _ = self.actions[action](bbox, self.alpha)
                if action == 'trigger':
                    reward = self.trigger_reward_function(predicted_bbox=new_bbox, ground_truth_box=ground_truth_bbox)
                else:
                    reward = self.movement_reward_function(previous_predicted_bbox=bbox, predicted_bbox=new_bbox,
                                                           ground_truth_box=ground_truth_bbox)
                if reward >= 0:
                    if reward == 3:
                        good_actions = [action]
                    else:
                        good_actions.append(action)
            if len(good_actions) != 0:
                next_move = random.choice(good_actions)
            else:
                next_move = random.choice(self.possible_actions)
        self.random_factor = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        return next_move

    def movement_reward_function(self, previous_predicted_bbox, predicted_bbox, ground_truth_box):
        return -1 if IoU(ground_truth_box, previous_predicted_bbox) <= IoU(ground_truth_box, predicted_bbox) else 1

    def trigger_reward_function(self, predicted_bbox, ground_truth_box):
        if IoU(ground_truth_box, predicted_bbox) >= self.IoU_treshold:
            return self.trigger_final_reward
        else:
            return -self.trigger_final_reward

    def get_state(self, clip_embedding):
        clip_embedding = clip_embedding.to(self.device)
        history_vector = torch.reshape(self.past_actions,
                                       (len(self.possible_actions) * self.maximum_past_actions_memory, 1)).to(
            self.device)
        state = torch.vstack((clip_embedding.T, history_vector))
        return state.to(self.device)

    def _encode_text(self, text):
        text_ = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_text(text_)

    def update_history_vector(self, action: str):
        if self.maximum_past_actions_memory <= 0:
            return
        if self._memory_update_index == 0:
            self.past_actions = torch.zeros(self.maximum_past_actions_memory * len(self.possible_actions))
        start = self._memory_update_index * len(self.possible_actions)
        end = start + len(self.possible_actions)
        self.past_actions[start:end] = torch.tensor(self.actions_dummies[action]).squeeze()
        if self._memory_update_index < self.maximum_past_actions_memory - 1:
            self._memory_update_index += 1
        else:
            self._memory_update_index = 0

    def _encode_img(self, image):
        image_ = self.clip_prep(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_image(image_)

    def encoder(self, raw_image, text):
        """ Produces an encoded tensor"""
        image = self._encode_img(raw_image)
        text = self._encode_text(text)

        encoding = torch.cat((image, text), 1)
        return encoding.to(device)


if __name__ == "__main__":
    from utilities import IoU

    agent = RL_Clip(maximum_past_actions_memory=10)
    print("Loading dataset")
    data_path = "/media/dmmp/vid+backup/Data/refcocog"
    # data_path = "dataset/refcocog"
    train_ds = RefCOCOg(ds_path=data_path, split='train')
    save_path = os.path.normpath(os.path.join("saved_models", "rl_clip"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # keep only a toy portion of each split
    batch_size = 128
    keep = 1
    train = False
    maximum_steps = 10

    if train:
        print("Instantiating model")
        dataset = RefCOCOg(ds_path=data_path, split='train')
        red_dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
        train_loader = torch.utils.data.DataLoader(red_dataset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=lambda x: x)
        del _

        device = 'cuda'
        epochs = 10
        GAMMA = 0.99
        TAU = 0.005
        memory_pointer = -1
        BB_LENGTH_LIMIT = 10  # if a BB is smaller than 10x10, the agent is penalized
        for epoch in tqdm(range(epochs)):
            epsilon = 1
            for step, batch in enumerate(train_loader):
                print("Batch " + str(step))
                print("Calculating batch embeddings")
                embeddings = []
                gt_bboxes = []
                initial_bboxes = []
                sentences = []
                images = []
                for element in batch:
                    for sentence in element['sentences']:
                        embeddings.append(agent.encoder(element['img'], sentence))
                        gt_bboxes.append(element['bbox'])
                        initial_bboxes.append([0, 0, element['img'].width, element[
                            'img'].height])  # get_image_center_bb(image_w=element['img'].width, image_h=element['img'].height))
                        sentences.append(sentence)
                        images.append(element['img'])
                print("Batch processing")
                for i, embedding in enumerate(embeddings):
                    bbox = gt_bboxes[i]
                    old_bbox = initial_bboxes[i]
                    image = images[i]
                    step = 0
                    reward = 0
                    masked = 0
                    not_finished = True
                    # status indicates whether the agent is still alive and has not triggered the terminal action
                    status = True
                    action = 0
                    input = agent.get_state(clip_embedding=embedding)
                    while status and (step < maximum_steps):
                        action = agent.choose_action(old_bbox, bbox)
                        new_bbox, status = agent.actions[action](old_bbox, agent.alpha)
                        if action != 'trigger':
                            reward = agent.movement_reward_function(predicted_bbox=new_bbox,
                                                                    previous_predicted_bbox=old_bbox,
                                                                    ground_truth_box=bbox)
                        else:
                            reward = agent.trigger_reward_function(predicted_bbox=new_bbox, ground_truth_box=bbox)
                        agent.update_history_vector(action)
                        step += 1

                        # special cases
                        try:
                            new_img = image.crop(new_bbox)
                        except PIL.Image.DecompressionBombError:
                            # the image is too big
                            new_img = image
                            reward = -agent.trigger_final_reward
                        # the image is too small
                        if ((new_bbox[2] - new_bbox[0]) < BB_LENGTH_LIMIT) or (
                                (new_bbox[3] - new_bbox[1]) < BB_LENGTH_LIMIT):
                            reward = -agent.trigger_final_reward
                        try:
                            encoding = agent.encoder(new_img, sentences[i])
                        except ZeroDivisionError:
                            encoding = agent.encoder(image, sentences[i])
                        new_input = agent.get_state(encoding)

                        # Experience replay storage
                        if len(agent.replay) < agent.buffer_experience_replay:
                            agent.replay.append((input, action, reward, new_input))
                        else:
                            # print("Starting memory replay")
                            agent.steps_done += 1
                            memory_pointer = memory_pointer + 1 if memory_pointer < agent.buffer_experience_replay - 1 else 0
                            agent.replay[memory_pointer] = (input, action, reward, new_input)

                            transitions = random.sample(agent.replay, batch_size)
                            minibatch = Transition(*zip(*transitions))
                            # we pick from the replay memory a sampled minibatch and generate the training samples

                            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                                    minibatch.next_state)), device=device,
                                                          dtype=torch.bool)
                            non_final_next_states = torch.squeeze(torch.stack(minibatch.next_state))
                            state_batch = torch.squeeze(torch.stack(minibatch.state)).to(device)
                            action_batch = torch.tensor(
                                [agent.actions_dummies[action] for action in minibatch.action]).to(
                                device)
                            reward_batch = torch.tensor(minibatch.reward).to(device)

                            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                            # columns of actions taken. These are the actions which would've been taken
                            # for each batch state according to policy_agent
                            q = agent.policy_net(state_batch)
                            state_action_values = q.gather(1, action_batch)[:, 1].squeeze()
                            next_state_values = torch.zeros(batch_size, device=device)
                            with torch.no_grad():
                                next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0]
                            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                            # Compute Huber loss
                            criterion = nn.SmoothL1Loss()
                            loss = criterion(state_action_values, expected_state_action_values)

                            # Optimize the model
                            agent.optimizer.zero_grad()
                            loss.backward()
                            # In-place gradient clipping
                            torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
                            agent.optimizer.step()
                            if agent.steps_done % 10 == 0:
                                # Soft update of the target network's weights
                                # θ′ ← τ θ + (1 −τ )θ′
                                target_net_state_dict = agent.target_net.state_dict()
                                policy_net_state_dict = agent.policy_net.state_dict()
                                for key in policy_net_state_dict:
                                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                                                 target_net_state_dict[
                                                                     key] * (1 - TAU)
                                agent.target_net.load_state_dict(target_net_state_dict)

                        input = new_input
                        old_bbox = new_bbox

                pass

            print("Saving epoch model")
            try:
                path = os.path.normpath(
                    os.path.join(save_path, "rl_clip_epoch_" + str(epoch) + "|" + str(loss.item())))
                with open(path, 'wb') as f:
                    pickle.dump(agent, f)
                    print("Model saved as: " + path)
            except NameError:
                pass

        print("Saving best model")
        files = os.listdir(save_path)
        losses = []
        for f in files:
            losses.append(f.split('|')[-1])
        best_loss_id = losses.index(min(losses))
        s = os.path.normpath(os.path.join(save_path, files[best_loss_id]))
        d = os.path.normpath(os.path.join(save_path, "best_model.pickle"))

        shutil.copyfile(src=s, dst=d)
        print(s + " has been saved as " + d)

    else:
        dataset = RefCOCOg(ds_path=data_path, split='test')
        red_dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
        test_loader = torch.utils.data.DataLoader(red_dataset, batch_size=batch_size, shuffle=False,
                                                  collate_fn=lambda x: x)
        print("samples: " + str(len(red_dataset)))
        print("Instantiating model")
        model_path = os.path.normpath(os.path.join(save_path, 'best_model.pickle'))
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)
        agent.batches = len(test_loader)

        # categories encoding
        print("Encoding categories")
        category_encodings = {}
        for category in dataset.categories.items():
            cat = category[1]['category']
            category_encodings[cat] = agent._encode_text("A picture of a " + cat)

        device = 'cuda'
        average_iou = 0
        iou = 0
        counter = 0

        avg_iou = 0.
        cum_iou = 0.

        cosine = 0
        avg_cosine = 0
        cum_cosine = 0

        euclidean_dist = 0
        avg_euclidean = 0
        cum_euclidean = 0

        grounding_acc = 0
        avg_grounding_acc = 0
        cum_grounding_acc = 0

        dot = 0
        avg_dot = 0
        cum_dot = 0

        j = 0
        for step, batch in tqdm(enumerate(test_loader)):

            images = []
            gt_bboxes = []
            initial_bboxes = []
            sentences = []
            categories = []
            for element in batch:
                for sentence in element['sentences']:
                    images.append(element['img'])
                    gt_bboxes.append(element['bbox'])
                    initial_bboxes.append([0, 0, element['img'].width, element[
                        'img'].height])  # get_image_center_bb(image_w=element['img'].width, image_h=element['img'].height))
                    sentences.append(sentence)
                    categories.append(element['category'])
            print("Predicting batch")
            for i, image in enumerate(images):
                # 10 step prediction
                step = 0
                status = True  # the agent hasn't finished
                ground_truth = gt_bboxes[i]

                new_bbox = initial_bboxes[i]
                while step < 10 and status:
                    step += 1
                    try:
                        embedding = agent.encoder(image.crop(new_bbox), sentences[i])
                    except ZeroDivisionError:
                        print(action)
                        print(new_bbox)
                        break
                    input = agent.get_state(clip_embedding=embedding).squeeze()
                    q = agent.policy_net.forward(input)
                    action = agent.possible_actions[torch.argmax(q)]
                    new_bbox, status = agent.actions[action](new_bbox, agent.alpha)

                iou = IoU(true_bbox=ground_truth, predicted_bbox=new_bbox)
                try:
                    text_enc = agent._encode_text(sentences[i]).float()
                    crop_enc = agent._encode_img(image.crop(new_bbox)).float()

                    cosine = cosine_similarity(crop_enc, text_enc).item()
                    euclidean_dist = torch.cdist(text_enc, crop_enc, p=2).squeeze().item()
                    dot = text_enc @ crop_enc.T
                    pass
                    grounding_acc = grounding_accuracy(crop_enc, category_encodings, true_category=categories[i])
                except ZeroDivisionError:
                    cosine = 0

                j += 1

                cum_iou += iou
                cum_cosine += cosine
                cum_euclidean += euclidean_dist
                cum_dot += dot[0,0].item()
                cum_grounding_acc += grounding_acc

            avg_iou = cum_iou / j
            avg_dot = cum_dot / j
            avg_cosine = cum_cosine / j
            avg_euclidean = cum_euclidean / j
            avg_grounding_acc = cum_grounding_acc / j

            print("Average IoU: " + str(avg_iou))
            print("Average Dot Product: " + str(avg_dot))
            print("Average Cosine: " + str(avg_cosine))
            print("Average Euclidean: " + str(avg_euclidean))
            print("Average Grounding_acc: " + str(avg_grounding_acc))
