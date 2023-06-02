# Algorithm 1 DiffusionDet Training
import math
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy
from torch.utils.data import random_split
from tqdm import tqdm

from refcocog import RefCOCOg, RefCOCOgSample
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from clip import clip

# strucure taken from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://arxiv.org/pdf/2208.04511.pdf

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        return self.layer3(x)


class RL_Clip(nn.Module):
    def __init__(self, actions=("left", 'right', 'top', 'down', 'bigger', 'smaller', 'fatter', 'taller', 'trigger'),
                 clip_ver="RN101", device='cuda',
                 maximum_past_actions_memory=10, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.actions = actions
        self.clip_model, self.clip_prep = clip.load(clip_ver, device=device)
        self.actions_history = []
        in_dim = 40  # clip output
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.IoU_treshold = 0.5
        self.trigger_final_reward = 3
        self.maximum_past_actions_memory = maximum_past_actions_memory
        self.past_actions = torch.zeros((1, len(actions) * self.maximum_past_actions_memory)).to(
            device)  # past action tensor, one-hot encoded


        self.policy_net = DQN(512 * 2 + len(self.past_actions), len(actions)).to(device)
        self.target_net = DQN(512 * 2 + len(self.past_actions), len(actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def movement_reward_function(self, previous_predicted_bbox, predicted_bbox, ground_truth_box):
        return numpy.sign(IoU(ground_truth_box, previous_predicted_bbox) - IoU(ground_truth_box, predicted_bbox))

    def trigger_reward_function(self, predicted_bbox, ground_truth_box):
        if IoU(ground_truth_box, predicted_bbox) >= self.IoU_treshold:
            return self.trigger_final_reward
        else:
            return -self.trigger_final_reward

    # αw = α(x2 − x1) αh = α(y2 − y1)
    def move_right(self, bbox, alpha):
        Aw = alpha * (bbox[2] - bbox[0])
        # Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0] + Aw, bbox[1], bbox[2] + Aw, bbox[3]]

    def move_left(self, bbox, alpha):
        Aw = alpha * (bbox[2] - bbox[0])
        # Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0] - Aw, bbox[1], bbox[2] - Aw, bbox[3]]

    def move_up(self, bbox, alpha):
        # Aw = alpha * (bbox[2] - bbox[0])
        Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0], bbox[1] - Ah, bbox[2], bbox[3] - Ah]

    def move_down(self, bbox, alpha):
        Aw = alpha * (bbox[2] - bbox[0])
        Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0], bbox[1] + Ah, bbox[2], bbox[3] + Ah]

    def make_bigger(self, bbox, alpha):
        Aw = alpha * (bbox[2] - bbox[0])
        Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0] - Aw, bbox[1] - Ah, bbox[2] + Aw, bbox[3] + Ah]

    def make_smaller(self, bbox, alpha):
        Aw = alpha * (bbox[2] - bbox[0])
        Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0] + Aw, bbox[1] + Ah, bbox[2] - Aw, bbox[3] - Ah]

    def make_fatter(self, bbox, alpha):
        # Aw = alpha * (bbox[2] - bbox[0])
        Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0], bbox[1] + Ah, bbox[2], bbox[3] - Ah]

    def make_taller(self, bbox, alpha):
        Aw = alpha * (bbox[2] - bbox[0])
        # Ah = alpha * (bbox[3] - bbox[1])

        return [bbox[0] + Aw, bbox[1], bbox[2] - Aw, bbox[3]]

    def training(self, data_loader):

        pass

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return None # torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # todo: capire cosa fare qui

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _encode_text(self, text):
        text_ = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_text(text_)

    def _encode_img(self, image):
        image_ = self.clip_prep(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.clip_model.encode_image(image_)

    def encoder(self, raw_image, text):
        """ Produces an encoded tensor"""
        image = self._encode_img(raw_image)
        text = self._encode_text(text)

        encoding = torch.cat((image, text), 1)
        return encoding


if __name__ == "__main__":
    from utilities import IoU

    print("Loading dataset")
    data_path = "/media/dmmp/vid+backup/Data/refcocog"
    # data_path = "dataset/refcocog"
    train_ds = RefCOCOg(ds_path=data_path, split='train')
    save_path = os.path.normpath(os.path.join("saved_models", "rl_clip"))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = RefCOCOg(ds_path=data_path)
    # keep only a toy portion of each split
    batch_size = 128
    keep = 0.01
    train = False
    red_dataset, _ = random_split(dataset, [int(keep * len(dataset)), len(dataset) - int(keep * len(dataset))])
    if train:
        print("Instantiating model")
        net = RL_Clip()
        red_train_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
        train_loader = torch.utils.data.DataLoader(red_train_ds, batch_size=batch_size, shuffle=True,
                                                   collate_fn=lambda x: x)
        del _


        device = 'cuda'
        epochs = 10
        for epoch in tqdm(range(epochs)):
            for step, batch in tqdm(enumerate(train_loader)):
                print("Extracting batch tensors", flush=True)
                batch_elements = []
                for el in batch:
                    for sentence in el['sentences']:
                        batch_elements.append(net.encoder(el['img'], sentence, el['bbox']))

                batch = torch.stack(batch_elements)
                print("Starting batch diffusion", flush=True)
                net.optimizer.zero_grad()

                batch_size = len(batch_elements)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, net.time, (batch_size,), device=device).long()

                loss = net.p_losses(batch, t, loss_type="huber")

                if step % 20 == 0:
                    print("Loss:", loss.item(), flush=True)

                loss.backward()
                net.optimizer.step()

            pass
            print("Saving epoch model")
            path = os.path.normpath(os.path.join(save_path, "diff_clip_epoch_" + str(epoch) + "|" + str(loss.item())))
            with open(path, 'wb') as f:
                pickle.dump(net, f)
                print("Model saved as: " + path)

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
        red_test_ds, _ = random_split(train_ds, [int(keep * len(train_ds)), len(train_ds) - int(keep * len(train_ds))])
        test_loader = torch.utils.data.DataLoader(red_test_ds, batch_size=batch_size, shuffle=True,
                                                  collate_fn=lambda x: x)
        print("Instantiating model")
        model_path = os.path.normpath(os.path.join(save_path, 'best_model.pickle'))
        with open(model_path, 'rb') as f:
            net = pickle.load(f)
        net.batches = len(test_loader)
        device = 'cuda'
        average_iou = 0
        iou = 0
        counter = 0
        true_bboxes = []
        for step, batch in tqdm(enumerate(test_loader)):
            print("Extracting batch tensors", flush=True)
            batch_elements = []
            for el in batch:
                for sentence in el['sentences']:
                    batch_elements.append(net.encoder(el['img'], sentence, None))
                    true_bboxes.append(el['bbox'])

            batch = torch.stack(batch_elements)
            print("Starting batch inference (t = " + str(net.time) + ")", flush=True)
            samples = net.sample(batch, image_size=40, batch_size=len(batch), channels=1)
            for i, sample in enumerate(samples):
                counter += 1
                predicted_bb = net.normalize_bbox(sample[0, 25:26, 24:28], reverse=True)
                true_bb = true_bboxes[i]
                iou += IoU(true_bb, predicted_bb)
                average_iou = iou / counter
                pass
            print("Average IoU: ", average_iou)
