from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from einops import rearrange

from .SMPL_H_specs import joints as effector_joints
from .specs import left_hand, right_hand, upper_body,\
    right_hand_chains, left_hand_chains, upper_body_chains


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim([z_middle - plot_radius, z_middle + plot_radius])


def plot_graph(x, y=None, title=None):
    if y is None:
        x, y = zip(*x)
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, fontsize=16)
    ax = fig.add_subplot(111)
    ax.plot(x, y)


class Plotter:
    def __init__(self):
        self.fig = plt.figure()
        self.ax_img = plt.subplot2grid((3, 9),        (0, 0), colspan=3, rowspan=3)
        self.ax_body_front = plt.subplot2grid((3, 9), (0, 3), projection='3d', colspan=2, )
        self.ax_body_top = plt.subplot2grid((3, 9),   (0, 5), projection='3d', colspan=2, )
        self.ax_body_proj = plt.subplot2grid((3, 9),  (0, 7), colspan=2, )
        self.ax_lh_front = plt.subplot2grid((3, 9),   (1, 3), projection='3d', colspan=2, )
        self.ax_lh_top = plt.subplot2grid((3, 9),     (1, 5), projection='3d', colspan=2, )
        self.ax_lh_proj = plt.subplot2grid((3, 9),    (1, 7), colspan=2, )
        self.ax_rh_front = plt.subplot2grid((3, 9),   (2, 3), projection='3d', colspan=2, )
        self.ax_rh_top = plt.subplot2grid((3, 9),     (2, 5), projection='3d', colspan=2, )
        self.ax_rh_proj = plt.subplot2grid((3, 9),    (2, 7), colspan=2, )

    @staticmethod
    def plot_3d(ax, joints, target_indexes=None, colour="red",  view='front'):
        ax.azim = -90
        if view == 'front':
            ax.elev = 90  # front on
        elif view == 'top':
            ax.elev = 180
        elif view == 'side':
            ax.elev = 0
            ax.azim = 135
        else:
            ax.azim, ax.elev = view

        plt.figure(1)
        joints = joints.clone().detach().cpu().numpy()
        if view == 'side':
            joints[0, :, [1, 2]] = joints[0, :, [2, 1]]

        if target_indexes is not None:
            joints = joints[target_indexes, :]
        ax.scatter(joints[0, :, 0], joints[0, :, 1], joints[0, :, 2], color=colour, )
        #ax.axis('off')
        set_axes_equal(ax)

    def plot_mesh(self, vertices, faces):
        vertices = vertices.detach().cpu().numpy().squeeze()
        if vertices.ndim > 2:
            mesh = Poly3DCollection(vertices[0, faces], alpha=0.1)
        else:
            mesh = Poly3DCollection(vertices[faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        self.ax_body_front.add_collection3d(mesh)
        if vertices.ndim > 2:
            mesh = Poly3DCollection(vertices[0, faces], alpha=0.1)
        else:
            mesh = Poly3DCollection(vertices[faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        self.ax_body_top.add_collection3d(mesh)

    @staticmethod
    def plot_2d(ax, keypoints, camera=None, target_indexes=None, colour="green", f=None):
        if camera is not None:
            keypoints = camera.project(keypoints, input_f=f)

        keypoints = keypoints.detach().cpu().numpy()

        if target_indexes is not None:
            keypoints = keypoints[0, target_indexes, :]
        ax.scatter(keypoints[0, :, 0], keypoints[0, :, 1], color=colour)

    @staticmethod
    def skeleton_2d(ax, keypoints, chain, camera=None, colour="green", f=None):
        if camera is not None:
            keypoints = camera.project(keypoints, input_f=f)
        keypoints = keypoints[0, [effector_joints[joint_name] for joint_name in chain]].detach().cpu().numpy()
        x, y, z = zip(*keypoints)
        ax.plot(x, y, color=colour)

    @staticmethod
    def skeleton_3d(ax, keypoints, chain, colour="red", label_effector=False):
        keypoints = keypoints[0, [effector_joints[joint_name] for joint_name in chain]].detach().cpu().numpy().squeeze()
        x, y, z = zip(*keypoints)
        ax.plot(x, y, z, color=colour)
        if label_effector:
            ax.scatter(x[-1], y[-1], z[-1], color="blue")

    @staticmethod
    def label(ax, joints):
        joints = joints.detach().cpu().numpy().squeeze()
        for index, (x, y, z) in enumerate(joints):
            ax.text(x, y, z, str(index), size=20, zorder=1)

    def plot_body_part(self, keypoints, target, camera, joint_names, axes, chains, f=None):
        ax_front, ax_top, ax_proj = axes
        target = rearrange(target, 'b xyz points -> b points xyz')
        upper_body_keypoints = keypoints[:, [effector_joints[joint_name] for joint_name in joint_names], :]
        upper_body_target = target[:, [effector_joints[joint_name] for joint_name in joint_names], :]
        self.plot_3d(ax_front, upper_body_keypoints, view='front')
        self.plot_3d(ax_top, upper_body_keypoints, view='top')
        self.plot_2d(ax_proj, upper_body_keypoints, camera, f=f)
        self.plot_2d(ax_proj, upper_body_target, colour="blue")
        for chain in chains:
            self.skeleton_2d(ax_proj, target, chain, colour="blue")
            self.skeleton_2d(ax_proj, keypoints, chain, camera=camera, f=f)
            self.skeleton_3d(ax_front, keypoints, chain)
            self.skeleton_3d(ax_top, keypoints, chain)

    def plot_upper_body(self, keypoints, target, camera, f=None):
        self.plot_body_part(keypoints, target, camera, upper_body,
                            (self.ax_body_front, self.ax_body_top, self.ax_body_proj), upper_body_chains, f=f)

    def plot_hands(self, keypoints, target, camera, f=None):
        self.plot_body_part(keypoints, target, camera, left_hand,
                            (self.ax_lh_front, self.ax_lh_top, self.ax_lh_proj), left_hand_chains, f=f)
        self.plot_body_part(keypoints, target, camera, right_hand,
                            (self.ax_rh_front, self.ax_rh_top, self.ax_rh_proj), right_hand_chains, f=f)

    @staticmethod
    def plot_image(ax, image, target_keypoints=None, target_indexes=None, label=False):
        image = image.astype(int)
        if len(image.shape) > 3:
            image = image[0]
        image = rearrange(image, 'c w h -> w h c')
        if target_keypoints is not None:
            target_keypoints = np.array(target_keypoints)
            if label:
                for index, (x, y, z) in enumerate(target_keypoints):
                    ax.text(x, y, str(index), size=20, zorder=1)
            if target_indexes is not None:
                target_keypoints = target_keypoints[list(target_indexes), :]
            ax.scatter(target_keypoints[:, 0], target_keypoints[:, 1], target_keypoints[:, 2], color="red")
        ax.imshow(image)

    def show_image(self, image, target_keypoints=None, target_indexes=None, label=False):
        self.plot_image(self.ax_img, image, target_keypoints, target_indexes, label)

    @staticmethod
    def target(target):
        fig = plt.figure()
        fig.suptitle('Target', fontsize=16)
        ax = fig.add_subplot(111,)
        Plotter.plot_2d(ax, target)

    @staticmethod
    def show(*args, **kwargs):
        plt.subplots_adjust(wspace=0.0,
                            hspace=0.0)
        return plt.show(*args, **kwargs)
        #plt.savefig('/home/joe/Pictures/2.png', dpi=500)

    @staticmethod
    def left_hand(keypoints):
        fig = plt.figure()
        fig.suptitle('Left hand', fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        left_hand_keypoints = keypoints[:, [effector_joints[joint_name] for joint_name in left_hand]]
        Plotter.plot_3d(ax, left_hand_keypoints, view='front')
        for chain in left_hand_chains:
            Plotter.skeleton_3d(ax, keypoints, chain)

    @staticmethod
    def right_hand(keypoints):
        fig = plt.figure()
        fig.suptitle('Right hand', fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        right_hand_keypoints = keypoints[:, [effector_joints[joint_name] for joint_name in right_hand]]
        Plotter.plot_3d(ax, right_hand_keypoints, view='front')
        for chain in right_hand_chains:
            Plotter.skeleton_3d(ax, keypoints, chain)

    @staticmethod
    def whole(keypoints, vertices, faces, label=False, fig=None, centre=9, size=0.5, view='front', name='Whole body'):
        if fig is None:
            fig = plt.figure()
        fig.suptitle(name, fontsize=16)
        ax = fig.add_subplot(111, projection='3d')
        Plotter.plot_3d(ax, keypoints, view=view)

        vertices = vertices.clone().detach().cpu().numpy().squeeze()

        if view == 'side':
            vertices[:, [1, 2]] = vertices[:, [2, 1]]

        if vertices.ndim > 2:
            mesh = Poly3DCollection(vertices[0, faces], alpha=0.1)
        else:
            mesh = Poly3DCollection(vertices[faces], alpha=0.1)
        face_color = (1.0, 1.0, 0.9)
        edge_color = (0, 0, 0)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        centre = keypoints[0, centre, :].detach().cpu().numpy()

        y, z = (2, 1) if view == 'side' else (1, 2)

        ax.set_xlim3d(centre[0] - size, centre[0] + size)
        ax.set_ylim3d(centre[y] - size, centre[y] + size)
        ax.set_zlim3d(centre[z] - size, centre[z] + size)
        #ax.dist = 5
        if label:
            Plotter.label(ax, keypoints)

    @staticmethod
    def save(filename):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0,)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        figs = [plt.figure(n) for n in plt.get_fignums()]
        for i, fig in enumerate(figs):
            fig.savefig(f"{filename}_{i}.png", format='png', bbox_inches='tight', pad_inches=0, dpi=1200)
            plt.close(fig)
            del fig
