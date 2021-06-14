from torch.optim import Adam
import torch
from IK.loss import euclidean_distance, layer_loss
from IK.utils.vis import plot_graph

def inverse_kinematics(model,
                       target,
                       target_indexes, camera,
                       confs=None,
                       a=0.5, b=0.5, c=0, d=0,
                       threshold=0.1,
                       max_repeats=1000,
                       suppress_output=False,
                       plot_loss=False):
    optimizer = Adam(
        [
            {"params": model.parameters(recurse=False), "lr": 0.1},
            {"params": camera.parameters(), "lr": 0.01},
        ],
        lr=0.001)

    if plot_loss:
        losses = []
        distances = []

    target = target.permute(0, 2, 1)

    for i in range(max_repeats):
        optimizer.zero_grad()
        joints_3d = model().joints
        projection = camera.project(joints_3d)

        distance = euclidean_distance(projection, target, target_indexes, confs=confs)
        loss = (a * distance) +\
               (b * layer_loss(joints_3d)) +\
               (c * model.head_loss()) +\
               (d * model.hand_loss())

        with torch.no_grad():
            if plot_loss:
                losses.append((i, loss.clone().detach().cpu().numpy()))
                distances.append((i, distance.clone().detach().cpu().numpy()))
            if distance < threshold:
                break

        loss.backward()
        optimizer.step()

        if not suppress_output and i % 20 == 0:
            print(i)

    if plot_loss and max_repeats > 0:
        plot_graph(losses, title="Loss")
        plot_graph(distances, title="Distance")
