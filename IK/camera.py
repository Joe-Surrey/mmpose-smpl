import torch.nn as nn
import torch

# 1, 0, 0  # width
# 0, 1, 0  # height
# 0, 0, 1  # depth

class Camera(nn.Module):
    def __init__(self):
        super(Camera, self).__init__()

    def project(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def project_single(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.project(input)


class OrthogonalCamera(Camera):
    def __init__(self, device="cpu"):
        super(OrthogonalCamera, self).__init__()

    def project_single(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(len(input))
        output[1:] = input[1:]
        return output

    def project(self, input):
        output = input
        output[:, 0, -1] = 0
        return output


class PerspectiveCamera(Camera):
    def __init__(self, device="cpu", fixed=False, f=torch.tensor([0.3])):#, p=[0, 0]):
        super(PerspectiveCamera, self).__init__()
        self.device = device

        self.f = f if fixed else nn.Parameter(f).to(self.device)

    def project_single(self, input: torch.Tensor) -> torch.Tensor:

        # Convert to homogenous
        point_4d = torch.ones((4, 1), dtype=torch.float32).to(self.device)
        point_4d[:3] = input[self.to_indexes].view(-1, 1)

        # Create camera matrix
        mat = torch.zeros((3, 4), dtype=torch.float32).to(self.device)
        mat[0, 0] = self.f
        mat[1, 1] = self.f
        mat[2, 2] = 1

        # Multiply
        projected_point = mat @ point_4d

        # Convert to 2d
        point_2d = (projected_point[self.from_indexes] / -projected_point[-1]).view(-1)
        point_2d[0] = 0

        return point_2d

    def project(self, input: torch.Tensor, input_f=None) -> torch.Tensor:
        if input_f is None:
            f = self.f
        else:
            f = input_f.unsqueeze(0)
        # Make homogonous
        point_4d = torch.nn.functional.pad(input, pad=(0, 1), mode='constant', value=1).unsqueeze(-1)

        # Create camera matrix
        bs, _, _ = input.shape
        mat = torch.zeros((bs, 1, 3, 4), dtype=torch.float32).to(self.device)
        mat[:, 0, 0, 0] = f
        mat[:, 0, 1, 1] = f
        mat[:, 0, 2, 2] = 1
        # Multiply
        projected_point = mat @ point_4d

        # Convert to 2d
        point_2d = projected_point.squeeze(-1) / -projected_point[:, :, -1]
        point_2d[:, :, 2] = 0

        return point_2d
