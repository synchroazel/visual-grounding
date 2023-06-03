# Algorithm 1 DiffusionDet Training
import math
import os
import pickle
import random
import shutil

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy
import numpy as np
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
                        ('state', 'action', 'reward','next_state', ))




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


def move_right(bbox, alpha: float) -> tuple:
    Aw = alpha * (bbox[2] - bbox[0])
    # Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0] + Aw, bbox[1], bbox[2] + Aw, bbox[3]], True


def move_left(bbox, alpha: float) -> tuple:
    Aw = alpha * (bbox[2] - bbox[0])
    # Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0] - Aw, bbox[1], bbox[2] - Aw, bbox[3]], True


def move_up(bbox, alpha: float) -> tuple:
    # Aw = alpha * (bbox[2] - bbox[0])
    Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0], bbox[1] - Ah, bbox[2], bbox[3] - Ah], True


def move_down(bbox, alpha: float) -> tuple:
    Aw = alpha * (bbox[2] - bbox[0])
    Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0], bbox[1] + Ah, bbox[2], bbox[3] + Ah], True


def make_bigger(bbox, alpha: float) -> tuple:
    Aw = alpha * (bbox[2] - bbox[0])
    Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0] - Aw, bbox[1] - Ah, bbox[2] + Aw, bbox[3] + Ah], True


def make_smaller(bbox, alpha: float) -> tuple:
    Aw = alpha * (bbox[2] - bbox[0])
    Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0] + Aw, bbox[1] + Ah, bbox[2] - Aw, bbox[3] - Ah], True


def make_fatter(bbox, alpha: float) -> tuple:
    # Aw = alpha * (bbox[2] - bbox[0])
    Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0], bbox[1] + Ah, bbox[2], bbox[3] - Ah], True


def make_taller(bbox, alpha: float) -> tuple:
    Aw = alpha * (bbox[2] - bbox[0])
    # Ah = alpha * (bbox[3] - bbox[1])

    return [bbox[0] + Aw, bbox[1], bbox[2] - Aw, bbox[3]], True


def stop(bbox, alpha):
    return bbox, False


class RL_Clip(nn.Module):
    def __init__(self, actions=("left", 'right', 'top', 'down', 'bigger', 'smaller', 'fatter', 'taller', 'trigger'),
                 clip_ver="RN101", device='cuda',
                 maximum_past_actions_memory=10, random_factor=0.2, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.possible_actions = actions
        self.actions_code = {"left":0, 'right':1, 'top':2, 'down':3, 'bigger':4, 'smaller':5, 'fatter':6, 'taller':7, 'trigger':8}
        self.actions = {"left": move_left, 'right': move_right, 'top': move_up, 'down': move_down,
                        'bigger': make_bigger,
                        'smaller': make_smaller, 'fatter': make_fatter, 'taller': make_taller, 'trigger': stop}
        self.clip_model, self.clip_prep = clip.load(clip_ver, device=device)
        self.history_vector = torch.zeros([len(actions) * maximum_past_actions_memory]).to(device)
        in_dim = 40  # clip output
        self.IoU_treshold = 0.5
        self.trigger_final_reward = 3
        self.maximum_past_actions_memory = maximum_past_actions_memory
        self.past_actions = torch.zeros((1, len(actions) * self.maximum_past_actions_memory)).to(
            device)  # past action tensor, one-hot encoded
        self.lr = 0.0001  # learning rate
        self.alpha = 0.5
        self.random_factor = random_factor
        self.buffer_exparience_replay = 1000

        # start the rewards table
        self.replay = []
        # G_state = G_state + α(target — G_state)

        self.policy_net = DQN(512 * 2 + len(actions) * self.maximum_past_actions_memory, len(actions)).to(device)
        self.target_net = DQN(512 * 2 +  len(actions) * self.maximum_past_actions_memory, len(actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        # self.steps_done = 0

    def init_reward(self, states):
        for action in states:
            self.G[action] = np.random.uniform(high=1.0, low=0.1)  # initialize with random values

    def choose_action(self, bbox, ground_truth_bbox):
        maxG = -10e15
        next_move = None
        randomN = np.random.random()
        if randomN < self.randomFactor:
            # if random number below random factor, choose random action
            next_move = np.random.choice(self.possible_actions)
        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            for action in self.possible_actions:
                new_bbox = self.actions[action](bbox, self.alpha)
                if action == 'trigger':
                    reward = self.trigger_reward_function(predicted_bbox=new_bbox, ground_truth_box=ground_truth_bbox)
                else:
                    reward = self.movement_reward_function(previous_predicted_bbox=bbox, predicted_bbox=new_bbox,
                                                           ground_truth_box=ground_truth_bbox)
                if reward >= maxG:
                    next_move = action
                    maxG = reward
        return next_move

    def movement_reward_function(self, previous_predicted_bbox, predicted_bbox, ground_truth_box):
        return numpy.sign(IoU(ground_truth_box, previous_predicted_bbox) - IoU(ground_truth_box, predicted_bbox))

    def trigger_reward_function(self, predicted_bbox, ground_truth_box):
        if IoU(ground_truth_box, predicted_bbox) >= self.IoU_treshold:
            return self.trigger_final_reward
        else:
            return -self.trigger_final_reward

    def update_history_vector(self, action: str):
        action_vector = torch.zeros(self.maximum_past_actions_memory)
        action_vector[self.possible_actions.index(action)] = 1
        size_history_vector = len(torch.nonzero(self.history_vector))
        number_of_actions = len(self.possible_actions)
        updated_history_vector = torch.zeros(self.maximum_past_actions_memory * number_of_actions)
        if size_history_vector < self.maximum_past_actions_memory:
            aux2 = 0
            for l in range(number_of_actions * size_history_vector,
                           number_of_actions * size_history_vector + number_of_actions - 1):
                self.history_vector[l] = action_vector[aux2]
                aux2 += 1
        else:
            for j in range(0, number_of_actions * (self.maximum_past_actions_memory - 1) - 1):
                updated_history_vector[j] = self.history_vector[j + number_of_actions]
            aux = 0
            for k in range(number_of_actions * (self.maximum_past_actions_memory - 1),
                           number_of_actions * self.maximum_past_actions_memory):
                updated_history_vector[k] = action_vector[aux]
                aux += 1
            self.history_vector = updated_history_vector

    def get_state(self, clip_embedding):
        clip_embedding = clip_embedding.to(self.device)
        history_vector = torch.reshape(self.history_vector, (len(self.possible_actions) * self.maximum_past_actions_memory, 1)).to(self.device)
        state = torch.vstack((clip_embedding.T, history_vector))
        return state.to(self.device)

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
        return encoding.to(device)


if __name__ == "__main__":
    from utilities import IoU

    agent = RL_Clip()
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
    train = True
    maximum_steps = 10
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
        epsilon = 0.1
        alpha = 0.3
        gamma = 0.9
        memory_pointer = -1
        TAU = 0.005
        for epoch in tqdm(range(epochs)):
            epsilon = 1
            for step, batch in enumerate(train_loader):
                print("Batch " + str(step))
                print("Calculating batch embeddings")
                embeddings = []
                gt_bboxes = []
                initial_bboxes = []
                sentences = []
                for element in batch:
                    for sentence in element['sentences']:
                        embeddings.append(net.encoder(element['img'], sentence))
                        gt_bboxes.append(element['bbox'])
                        initial_bboxes.append([0, 0, element['img'].width, element['img'].height])
                        sentences.append(sentence)

                for i, embedding in tqdm(enumerate(embeddings)):
                    bbox = gt_bboxes[i]
                    old_bbox = initial_bboxes[i]
                    step = 0
                    reward = 0
                    masked = 0
                    not_finished = True
                    # status indicates whether the agent is still alive and has not triggered the terminal action
                    status = True
                    action = 0
                    input = net.get_state(clip_embedding=embedding)
                    while status and (step < maximum_steps):
                        qval = net.policy_net.forward(input.T)
                        if epsilon > random.random():
                            action = random.choice(net.possible_actions)
                            epsilon += 0.01
                        else:
                            action = net.possible_actions[torch.argmax(qval).item()]
                        new_bbox, status = net.actions[action](old_bbox, alpha)
                        if action != 'trigger':
                            reward = net.movement_reward_function(predicted_bbox=new_bbox,
                                                                  previous_predicted_bbox=old_bbox,
                                                                  ground_truth_box=bbox)
                        else:
                            reward = net.trigger_reward_function(predicted_bbox=new_bbox, ground_truth_box=bbox)
                        net.update_history_vector(action)
                        step += 1

                        # get new features
                        try:
                            new_img = batch[i]['img'].crop(new_bbox)

                        except IndexError:
                                break
                        try:
                            encoding = net.encoder(new_img, sentences[i])
                        except ZeroDivisionError:
                            encoding = net.encoder(batch[i]['img'], sentences[i])
                        new_input = net.get_state(encoding)

                        # Experience replay storage
                        if len(net.replay) < net.buffer_exparience_replay:
                            net.replay.append((input, action, reward, new_input))
                        else:
                            print("Starting memory replay")
                            memory_pointer = memory_pointer + 1 if memory_pointer < net.buffer_exparience_replay else 0
                            net.replay[memory_pointer] = (input, action, reward, new_input)

                            transitions = random.sample(net.replay, batch_size)
                            minibatch = Transition(*zip(*transitions))
                            # we pick from the replay memory a sampled minibatch and generate the training samples

                            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                                    minibatch.next_state)), device=device,
                                                          dtype=torch.bool)
                            non_final_next_states = torch.cat([s for s in minibatch.next_state
                                                               if s is not None])
                            state_batch = torch.squeeze(torch.stack(minibatch.state)).to(device)
                            action_batch = torch.tensor([net.actions_code[action] for action in minibatch.action]).to(device)
                            reward_batch = torch.tensor(minibatch.reward).to(device)

                            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                            # columns of actions taken. These are the actions which would've been taken
                            # for each batch state according to policy_net
                            q = net.policy_net(state_batch)
                            state_action_values = q.gather(0, action_batch)
                            next_state_values = torch.zeros(batch_size, device=device)
                            with torch.no_grad():
                                next_state_values[non_final_mask] = net.target_net(non_final_next_states).max(1)[0]
                            expected_state_action_values = (next_state_values * gamma) + reward_batch
                            # Compute Huber loss
                            criterion = nn.SmoothL1Loss()
                            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                            # Optimize the model
                            net.optimizer.zero_grad()
                            loss.backward()
                            # In-place gradient clipping
                            torch.nn.utils.clip_grad_value_(net.policy_net.parameters(), 100)
                            net.optimizer.step()



                            pass
                            # hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=0)
                        input = new_input
                        old_bbox = new_bbox

                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        target_net_state_dict = net.target_net.state_dict()
                        policy_net_state_dict = net.policy_net.state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[
                                key] * (1 - TAU)
                        net.target_net.load_state_dict(target_net_state_dict)

                    net.optimizer.zero_grad()

                pass
                print("Saving epoch model")
                try:
                    path = os.path.normpath(
                        os.path.join(save_path, "diff_clip_epoch_" + str(epoch) + "|" + str(loss.item())))
                    with open(path, 'wb') as f:
                        pickle.dump(net, f)
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
