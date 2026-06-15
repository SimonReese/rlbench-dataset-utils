import sys
import pickle
from os import listdir
from os.path import exists, join
from typing import List

import numpy as np
from natsort import natsorted
from PIL import Image
from scipy.spatial.transform import Rotation

from .demo import Demo
from .observation import Observation
from .observation_config import ObservationConfig
from .rlbench_const import *
from .vision_sensor import VisionSensor

from . import demo
from . import observation
import rlbench_dataset_utils

# Patch rlbench references in the pickle files. Will overwrite rlbench installation so avoid install it if rbench is already installed
sys.modules["rlbench"] = rlbench_dataset_utils
sys.modules["rlbench.demo"] = demo
sys.modules["rlbench.backend.observation"] = observation

def image_to_float_array(image, scale_factor=None):
    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

    Returns:
    A 2D floating point numpy array representing a depth image.

    """
    DEFAULT_RGB_SCALE_FACTOR = 256000.0
    DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                             np.uint16: 1000.0,
                             np.int32: DEFAULT_RGB_SCALE_FACTOR}

    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array


def rgb_handles_to_mask(rgb_coded_handles:np.ndarray):
    # rgb_coded_handles should be (w, h, c)
    # Handle encoded as : handle = R + G * 256 + B * 256 * 256

    # If reading from a PIL image, the values will already be on the 
    # range [0, 255] and uint8. However, if rgb_coded_handles is read directly from 
    # a sensor, it will be on the range [0, 1].
    if rgb_coded_handles.dtype != np.uint8:
        rgb_coded_handles *= 255  # takes rgb range to 0 -> 255
    rgb_coded_handles = rgb_coded_handles.astype(np.uint32)
    return (rgb_coded_handles[:, :, 0] +
            rgb_coded_handles[:, :, 1] * 256 +
            rgb_coded_handles[:, :, 2] * 256 * 256)

# Get list of variations for task
def get_variations_ids(dataset_path: str, task_name:str) -> List[int]:
    """Retuns a list of indices of variations for a given task in a stored dataset
    
        :param str dataset_path: path to RLBench dataset root folder
        :param str task_name: name of task to consider in the dataset
        
        :return List[int]: a list of indices, one for each stored variation
        
        :raises RuntimeError: if something is wrong with indices in the folder
    """
    # Open task folder
    variation_folders = listdir(join(dataset_path, task_name))
    variation_ids = []
    id: int = 0
    while len(variation_ids) != len(variation_folders):
        if VARIATIONS_FOLDER % id in variation_folders:
            variation_ids.append(id)
        id += 1

        # Make sure we have a good dataset
        if id >= len(variation_folders) * 2: raise RuntimeError(
            f"Error while loading variation ids for task {task_name} in {dataset_path}:\n ",
            f"The folder contains {len(variation_folders)} folders but only {len(variation_ids)} ids were found.\n",
            f"Nevertheless we were lookin up to id {id} which is 2*#folders in the task folder.\n",
            f"Someting feels wrong. Please check dataset integrity.\n",
            f"Discovered ids: {variation_ids}\nVariation folders: {variation_folders}")
    return variation_ids

# get_stored_demos requires patch to load this module instead of full rlbench
def get_stored_demos(amount: int, image_paths: bool, dataset_root: str,
                     variation_number: int, task_name: str,
                     obs_config: ObservationConfig,
                     random_selection: bool = True,
                     from_episode_number: int = 0) -> List[Demo]:
    """ Returns an episode as a list of observations from stored dataset for a given task and variation

        :param amount: how many examples to extract. Set -1 to extract all obs. Throws RuntimeError if not enough avaialble
        :param image_paths: if False, images will be stored as array inside the observations. Otherwise, the observations will contain the path
        :param dataset_root: path to the root folder of the dataset. It will be the one where ROOT/task_name0/variation0/..., ROOT/task_name1 ...
        :param variation_number: the index of the variation to retreive
        :param task_name: name of the task to retreive
        :param obs_config: the observation configuration. Images will be resized if needed, but only if stored in obs dict
        :param random_selection: if true will retrive an amount of random example episodes
        :param from_episode_number: if random selection is false, will retrieve amount episodes starting from episode from_episode_number

        :returns: a list of Demo, one Demo for each episode. The episodes will be sorted

        :raise: RuntimeError if not enough obs avaialble
    """

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    # Sample an amount of examples for the variation of this task
    examples_path = join(
        task_root, VARIATIONS_FOLDER % variation_number,
        EPISODES_FOLDER)
    examples = listdir(examples_path)
    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f) # This will try to load module rlench.demo. we need to trick it with the patch at the beginnign of file

        l_sh_rgb_f = join(example_path, LEFT_SHOULDER_RGB_FOLDER)
        l_sh_depth_f = join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
        l_sh_mask_f = join(example_path, LEFT_SHOULDER_MASK_FOLDER)
        r_sh_rgb_f = join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
        r_sh_depth_f = join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
        r_sh_mask_f = join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
        oh_rgb_f = join(example_path, OVERHEAD_RGB_FOLDER)
        oh_depth_f = join(example_path, OVERHEAD_DEPTH_FOLDER)
        oh_mask_f = join(example_path, OVERHEAD_MASK_FOLDER)
        wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
        wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
        wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
        front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
        front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
        front_mask_f = join(example_path, FRONT_MASK_FOLDER)

        num_steps = len(obs)

        # Checks integrty of dataset (number of steps must be the same number of images in each image folder)
        if not (num_steps == len(listdir(l_sh_rgb_f)) == len(
                listdir(l_sh_depth_f)) == len(listdir(r_sh_rgb_f)) == len(
                listdir(r_sh_depth_f)) == len(listdir(oh_rgb_f)) == len(
                listdir(oh_depth_f)) == len(listdir(wrist_rgb_f)) == len(
                listdir(wrist_depth_f)) == len(listdir(front_rgb_f)) == len(
                listdir(front_depth_f))):
            raise RuntimeError('Broken dataset assumption')

        # For every step, store path to corresponding image in the observation dictionary
        for i in range(num_steps):
            si = IMAGE_FORMAT % i
            if obs_config.left_shoulder_camera.rgb:
                obs[i].left_shoulder_rgb = join(l_sh_rgb_f, si)
            if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                obs[i].left_shoulder_depth = join(l_sh_depth_f, si)
            if obs_config.left_shoulder_camera.mask:
                obs[i].left_shoulder_mask = join(l_sh_mask_f, si)
            if obs_config.right_shoulder_camera.rgb:
                obs[i].right_shoulder_rgb = join(r_sh_rgb_f, si)
            if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                obs[i].right_shoulder_depth = join(r_sh_depth_f, si)
            if obs_config.right_shoulder_camera.mask:
                obs[i].right_shoulder_mask = join(r_sh_mask_f, si)
            if obs_config.overhead_camera.rgb:
                obs[i].overhead_rgb = join(oh_rgb_f, si)
            if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                obs[i].overhead_depth = join(oh_depth_f, si)
            if obs_config.overhead_camera.mask:
                obs[i].overhead_mask = join(oh_mask_f, si)
            if obs_config.wrist_camera.rgb:
                obs[i].wrist_rgb = join(wrist_rgb_f, si)
            if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                obs[i].wrist_depth = join(wrist_depth_f, si)
            if obs_config.wrist_camera.mask:
                obs[i].wrist_mask = join(wrist_mask_f, si)
            if obs_config.front_camera.rgb:
                obs[i].front_rgb = join(front_rgb_f, si)
            if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                obs[i].front_depth = join(front_depth_f, si)
            if obs_config.front_camera.mask:
                obs[i].front_mask = join(front_mask_f, si)

            # Remove low dim info if necessary
            if not obs_config.joint_velocities:
                obs[i].joint_velocities = None
            if not obs_config.joint_positions:
                obs[i].joint_positions = None
            if not obs_config.joint_forces:
                obs[i].joint_forces = None
            if not obs_config.gripper_open:
                obs[i].gripper_open = None
            if not obs_config.gripper_pose:
                obs[i].gripper_pose = None
            if not obs_config.gripper_joint_positions:
                obs[i].gripper_joint_positions = None
            if not obs_config.gripper_touch_forces:
                obs[i].gripper_touch_forces = None
            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None

        # If we need to have images as array in observation dictionary (and not path), we load each array image in dictionary
        if not image_paths:
            for i in range(num_steps):
                if obs_config.left_shoulder_camera.rgb:
                    obs[i].left_shoulder_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_rgb),
                            obs_config.left_shoulder_camera.image_size))
                if obs_config.right_shoulder_camera.rgb:
                    obs[i].right_shoulder_rgb = np.array(
                        _resize_if_needed(Image.open(
                        obs[i].right_shoulder_rgb),
                            obs_config.right_shoulder_camera.image_size))
                if obs_config.overhead_camera.rgb:
                    obs[i].overhead_rgb = np.array(
                        _resize_if_needed(Image.open(
                        obs[i].overhead_rgb),
                            obs_config.overhead_camera.image_size))
                if obs_config.wrist_camera.rgb:
                    obs[i].wrist_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_rgb),
                            obs_config.wrist_camera.image_size))
                if obs_config.front_camera.rgb:
                    obs[i].front_rgb = np.array(
                        _resize_if_needed(
                            Image.open(obs[i].front_rgb),
                            obs_config.front_camera.image_size))

                if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
                    l_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].left_shoulder_depth),
                            obs_config.left_shoulder_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['left_shoulder_camera_near']
                    far = obs[i].misc['left_shoulder_camera_far']
                    l_sh_depth_m = near + l_sh_depth * (far - near)
                    if obs_config.left_shoulder_camera.depth:
                        d = l_sh_depth_m if obs_config.left_shoulder_camera.depth_in_meters else l_sh_depth
                        obs[i].left_shoulder_depth = obs_config.left_shoulder_camera.depth_noise.apply(d)
                    else:
                        obs[i].left_shoulder_depth = None

                if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
                    r_sh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].right_shoulder_depth),
                            obs_config.right_shoulder_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['right_shoulder_camera_near']
                    far = obs[i].misc['right_shoulder_camera_far']
                    r_sh_depth_m = near + r_sh_depth * (far - near)
                    if obs_config.right_shoulder_camera.depth:
                        d = r_sh_depth_m if obs_config.right_shoulder_camera.depth_in_meters else r_sh_depth
                        obs[i].right_shoulder_depth = obs_config.right_shoulder_camera.depth_noise.apply(d)
                    else:
                        obs[i].right_shoulder_depth = None

                if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
                    oh_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].overhead_depth),
                            obs_config.overhead_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['overhead_camera_near']
                    far = obs[i].misc['overhead_camera_far']
                    oh_depth_m = near + oh_depth * (far - near)
                    if obs_config.overhead_camera.depth:
                        d = oh_depth_m if obs_config.overhead_camera.depth_in_meters else oh_depth
                        obs[i].overhead_depth = obs_config.overhead_camera.depth_noise.apply(d)
                    else:
                        obs[i].overhead_depth = None

                if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
                    wrist_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].wrist_depth),
                            obs_config.wrist_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['wrist_camera_near']
                    far = obs[i].misc['wrist_camera_far']
                    wrist_depth_m = near + wrist_depth * (far - near)
                    if obs_config.wrist_camera.depth:
                        d = wrist_depth_m if obs_config.wrist_camera.depth_in_meters else wrist_depth
                        obs[i].wrist_depth = obs_config.wrist_camera.depth_noise.apply(d)
                    else:
                        obs[i].wrist_depth = None

                if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
                    front_depth = image_to_float_array(
                        _resize_if_needed(
                            Image.open(obs[i].front_depth),
                            obs_config.front_camera.image_size),
                        DEPTH_SCALE)
                    near = obs[i].misc['front_camera_near']
                    far = obs[i].misc['front_camera_far']
                    front_depth_m = near + front_depth * (far - near)
                    if obs_config.front_camera.depth:
                        d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth
                        obs[i].front_depth = obs_config.front_camera.depth_noise.apply(d)
                    else:
                        obs[i].front_depth = None

                if obs_config.left_shoulder_camera.point_cloud:
                    obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        l_sh_depth_m, # pyright: ignore[reportPossiblyUnboundVariable]
                        obs[i].misc['left_shoulder_camera_extrinsics'],
                        obs[i].misc['left_shoulder_camera_intrinsics'])
                if obs_config.right_shoulder_camera.point_cloud:
                    obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        r_sh_depth_m, # pyright: ignore[reportPossiblyUnboundVariable]
                        obs[i].misc['right_shoulder_camera_extrinsics'],
                        obs[i].misc['right_shoulder_camera_intrinsics'])
                if obs_config.overhead_camera.point_cloud:
                    obs[i].overhead_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        oh_depth_m, # pyright: ignore[reportPossiblyUnboundVariable]
                        obs[i].misc['overhead_camera_extrinsics'],
                        obs[i].misc['overhead_camera_intrinsics'])
                if obs_config.wrist_camera.point_cloud:
                    obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        wrist_depth_m, # pyright: ignore[reportPossiblyUnboundVariable]
                        obs[i].misc['wrist_camera_extrinsics'],
                        obs[i].misc['wrist_camera_intrinsics'])
                if obs_config.front_camera.point_cloud:
                    obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                        front_depth_m, # pyright: ignore[reportPossiblyUnboundVariable]
                        obs[i].misc['front_camera_extrinsics'],
                        obs[i].misc['front_camera_intrinsics'])

                # Masks are stored as coded RGB images.
                # Here we transform them into 1 channel handles.
                if obs_config.left_shoulder_camera.mask:
                    obs[i].left_shoulder_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].left_shoulder_mask),
                            obs_config.left_shoulder_camera.image_size)))
                if obs_config.right_shoulder_camera.mask:
                    obs[i].right_shoulder_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].right_shoulder_mask),
                            obs_config.right_shoulder_camera.image_size)))
                if obs_config.overhead_camera.mask:
                    obs[i].overhead_mask = rgb_handles_to_mask(
                        np.array(_resize_if_needed(Image.open(
                            obs[i].overhead_mask),
                            obs_config.overhead_camera.image_size)))
                if obs_config.wrist_camera.mask:
                    obs[i].wrist_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].wrist_mask),
                            obs_config.wrist_camera.image_size)))
                if obs_config.front_camera.mask:
                    obs[i].front_mask = rgb_handles_to_mask(np.array(
                        _resize_if_needed(Image.open(
                            obs[i].front_mask),
                            obs_config.front_camera.image_size)))

        demos.append(obs)
    return demos

def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image

def get_panda_gripper_open_amount(gripper_joint_positions: np.ndarray) -> List[float]:
        """Gets the gripper open state for the panda gripper. 1 means open, whilst 0 means closed.

        PANDA_JOINT_INTERVALS_LIST = [
            [0.0, 0.03999999910593033],
            [0.0, 0.03999999910593033]
        ]

        :param gripper_joint_positions: numpy.ndarray containing the current position of the gripper joints

        :return: A list of floats between 0 and 1 representing the gripper open
            state for each joint. 1 means open, whilst 0 means closed.
        """
        PANDA_JOINT_INTERVALS_LIST = [[0.0, 0.03999999910593033], [0.0, 0.03999999910593033]]
        joint_intervals_list = PANDA_JOINT_INTERVALS_LIST
        joint_intervals = np.array(joint_intervals_list)
        joint_range = joint_intervals[:, 1] - joint_intervals[:, 0]
        return list(np.clip((np.array(
            gripper_joint_positions) - joint_intervals[:, 0]) /
                            joint_range, 0.0, 1.0))


# ------- SPATIAL POSE UTILITY FUNCTIONS ------------
def pose_to_T(p, q):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = p
    return T

def invert_T(T):
    Rm = T[:3, :3]
    p = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ p
    return T_inv

def delta_pose_ee(p_cur, q_cur, p_des, q_des):
    """
    Returns delta pose in EE frame as:
    [dx, dy, dz, qx, qy, qz, qw]
    """
    T_cur = pose_to_T(p_cur, q_cur)
    T_des = pose_to_T(p_des, q_des)

    T_delta = invert_T(T_cur) @ T_des

    p_delta = T_delta[:3, 3]
    q_delta = Rotation.from_matrix(T_delta[:3, :3]).as_quat()

    return np.hstack((p_delta, q_delta))

def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(quaternion)
    angles = rotation.as_euler("xyz")
    return angles

def euler_to_quaternion(euler: np.ndarray, format: str = "xyz") -> np.ndarray:
    rotation = Rotation.from_euler(format, euler)
    quaternion = rotation.as_quat()
    return quaternion

