from torch.optim import Adam
import torch
from IK.loss import reprojection_loss, hand_reprojection_loss
from IK.utils.vis import Plotter


def fit(model, camera, dataloader, reprojection_model=None, batch_size=1, suppress_output=False, eval_step=100, learn_camera=False):

    optimiser = Adam(
            [
                {"params": model.parameters(), "lr": 0.001},
                {"params": camera.parameters(), "lr": 0.00001},
            ],
            lr=0.000001)

    best_loss = float("inf")
    iters = 0
    meta_iters = 0
    running_loss = 0

    loss_func = reprojection_loss(camera=camera, batch_size=batch_size, model=reprojection_model, learn_camera=learn_camera)
    for epoch in range(100):

        for (image, target, target_indexes, confidences) in dataloader:
            model.train()
            camera.train()

            optimiser.zero_grad()
            angles = model(image)

            loss = loss_func(angles, target, target_indexes, confidences)

            loss.backward()
            optimiser.step()

            iters += 1

            with torch.no_grad():
                running_loss += loss

                f = torch.abs(angles[:, -1].detach()) if learn_camera else None
                if not suppress_output and iters == eval_step:

                    model.eval()
                    camera.eval()

                    running_loss = running_loss / eval_step
                    if running_loss < best_loss:
                        best_loss = running_loss
                        if not learn_camera:
                            torch.save(camera.state_dict(), "/vol/research/SignRecognition/smplx/ckpts/best_camera.pt")
                        torch.save(model.state_dict(), "/vol/research/SignRecognition/smplx/ckpts/best_model.pt")

                    print(f"Epoch: {epoch} iter: {meta_iters} Running loss: {running_loss}")

                    if reprojection_model is not None:
                        output = reprojection_model(angles, return_verts=True)

                        plt = Plotter()
                        plt.show_image(image.detach().numpy())  # , label=True)

                        plt.plot_hands(output.joints, target, camera=camera, f=f)
                        plt.plot_upper_body(output.joints, target, camera=camera, f=f)

                        plt.plot_mesh(output.vertices, reprojection_model.faces)
                        plt.whole(output.joints, output.vertices, reprojection_model.faces)
                        Plotter.save(f"/vol/research/SignRecognition/swisstxt/outs/{epoch}_{meta_iters}")
                        meta_iters += 1
                    running_loss = 0
                    iters = 0


def fit_hands(model, camera, dataloader, reprojection_model=None, batch_size=1, suppress_output=False, eval_step=100, learn_camera=False):

    optimiser = Adam(
            [
                {"params": model.parameters(), "lr": 0.001},
            ],
            lr=0.000001)

    best_loss = float("inf")
    iters = 0
    meta_iters = 0
    running_loss = 0

    loss_func = hand_reprojection_loss(camera=camera, batch_size=batch_size, model=reprojection_model, learn_camera=learn_camera)
    for epoch in range(100):

        for (image, target, target_indexes, confidences, transl, body_pose, f) in dataloader:
            model.train()
            camera.train()

            optimiser.zero_grad()
            angles = model(image)

            loss = loss_func(angles, target, target_indexes, confidences, transl, body_pose, f)

            loss.backward()
            optimiser.step()

            iters += 1

            with torch.no_grad():
                running_loss += loss

                if not suppress_output and iters == eval_step:

                    model.eval()
                    camera.eval()

                    running_loss = running_loss / eval_step
                    if running_loss < best_loss:
                        best_loss = running_loss
                        if not learn_camera:
                            torch.save(camera.state_dict(), "/vol/research/SignRecognition/smplx/ckpts/best_hand_camera.pt")
                        torch.save(model.state_dict(), "/vol/research/SignRecognition/smplx/ckpts/best_hand_model.pt")

                    print(f"Epoch: {epoch} iter: {meta_iters} Running loss: {running_loss}")

                    if reprojection_model is not None:
                        output = reprojection_model(angles, transl, body_pose, return_verts=True)

                        plt = Plotter()
                        plt.show_image(image.detach().numpy())  # , label=True)

                        plt.plot_hands(output.joints, target, camera=camera, f=f)
                        plt.plot_upper_body(output.joints, target, camera=camera, f=f)

                        plt.plot_mesh(output.vertices, reprojection_model.faces)
                        plt.whole(output.joints, output.vertices, reprojection_model.faces)
                        Plotter.save(f"/vol/research/SignRecognition/swisstxt/hand_outs/{epoch}_{meta_iters}")
                        meta_iters += 1
                    running_loss = 0
                    iters = 0
