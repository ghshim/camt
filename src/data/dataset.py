import numpy as np
import pickle
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from src.utils.data_utils import load_data
from src.utils.constants import NUM_JOINTS
from src.data.sentence_transformer import encode


class MotionDataset(Dataset):
    def __init__(
            self, 
            frame_lists, 
            initial_poses, 
            next_motions, 
            labels, 
            object_boxes, 
            device,
            normalize=False
        ):
        self.frame_lists = frame_lists

        self.initial_poses = initial_poses
        self.next_motions = next_motions
        self.labels = labels
        # sentence encoding using SentenceTransformer
        self.labels_embeddings = self.encode_labels() 
        
        self.object_boxes = object_boxes   
        
        self.device = device     
        self.normalize = normalize

        print("initial_poses", initial_poses[0].shape)
        print("next_motions", next_motions[0].shape)
        print("labels_embeddings", self.labels_embeddings[0].shape)
        print("labels", labels[0])
        print("object_boxes", object_boxes[0].shape)

        if normalize:
            self.mean_pose, self.std_pose = None, None
            self.mean_gvel, self.std_gvel = None, None
            self.mean_rvel, self.std_rvel = None, None
            standardized_poses, standardized_global_velocity, standardized_rotational_velocity = self.standardize_data(self.initial_poses)
    
    def __getitem__(self, index):
        """Returns one data pair (input data, output data)."""
        frame_list = self.frame_lists[index]
        label = self.labels[index]

        initial_pose = torch.Tensor(self.initial_poses[index]).to(self.device) # (1, 72)
        next_motion = torch.Tensor(self.next_motions[index]).to(self.device) # (NUM_SEQ, 76)
        motion_descr = torch.Tensor(self.labels_embeddings[index]).to(self.device) # (384,)
        object_box = torch.Tensor(self.object_boxes[index]).to(self.device)
            
        # print('initial_pose.shape', initial_pose.shape)
        # print('object_box.shape', object_box.shape)
        # print('motion_description.shape', motion_description.shape)
        # print('next_motion.shape:', next_motion.shape)
        # print('------------------------')
        return frame_list, label, initial_pose, next_motion, motion_descr, object_box
    
    def __len__(self):
        return len(self.initial_poses)

    def encode_labels(self):
        embeddings = encode(self.labels)
        return embeddings

    def parse_motion_vector(motion_vector):
        num_frames_minus_1, _ = motion_vector.shape
        num_frames = num_frames_minus_1 + 1
        
        poses = np.zeros((num_frames, NUM_JOINTS, 3))
        poses[1:] = motion_vector[:, :NUM_JOINTS*3].reshape(num_frames_minus_1, NUM_JOINTS, 3)
        
        translation = np.zeros((num_frames, 3))
        translation[1:] = motion_vector[:, NUM_JOINTS*3:NUM_JOINTS*3+3]
        
        forward_direction = np.zeros((num_frames, 3))
        forward_direction[1:] = motion_vector[:, NUM_JOINTS*3+3:NUM_JOINTS*3+6]
        
        global_velocity = np.zeros((num_frames_minus_1, 3))
        global_velocity[:] = motion_vector[:, NUM_JOINTS*3+6:NUM_JOINTS*3+9]
        
        rotational_velocity = np.zeros((num_frames_minus_1, 1))
        rotational_velocity[:] = motion_vector[:, NUM_JOINTS*3+9:]
        
        return poses, translation, forward_direction, global_velocity, rotational_velocity
    
    def standardize_data(self, initial_poses):
        poses, _, _, global_velocity, rotational_velocity = self.parse_motion_vector(initial_poses)
        (self.mean_pose, self.std_pose), (self.mean_gvel, self.std_gvel), (self.mean_rvel, self.std_rvel)  \
                                              = self.calculate_mean_std(poses, global_velocity, rotational_velocity)
        poses_flattened = poses.reshape(-1, 3)
        standardized_poses = (poses_flattened - self.mean_pose) / self.std_pose
        standardized_poses = standardized_poses.reshape(poses.shape)
        
        standardized_global_velocity = (global_velocity - self.mean_gvel) / self.std_gvel
        standardized_rotational_velocity = (rotational_velocity - self.mean_rvel) / self.std_rvel
        
        return standardized_poses, standardized_global_velocity, standardized_rotational_velocity
        
    def calculate_mean_std(self, poses, global_velocity, rotational_velocity):
        poses_flattened = poses.reshape(-1, 3)  # Flatten poses to (NUM_FRAMES-1 * JOINT_NUM, 3)
        
        mean_pose = np.mean(poses_flattened, axis=0)
        std_pose = np.std(poses_flattened, axis=0)
        
        mean_global_velocity = np.mean(global_velocity, axis=0)
        std_global_velocity = np.std(global_velocity, axis=0)
        
        mean_rotational_velocity = np.mean(rotational_velocity, axis=0)
        std_rotational_velocity = np.std(rotational_velocity, axis=0)
        
        return (mean_pose, std_pose), (mean_global_velocity, std_global_velocity), (mean_rotational_velocity, std_rotational_velocity)


def create_datasets(frame_lists, initial_poses, next_motions, labels, object_boxes, device, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2):
    dataset = MotionDataset(frame_lists, initial_poses, next_motions, labels, object_boxes, device)
    dataset_size = len(dataset)

    # split train, val, test
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset


def get_loader(
    dataset_path,
    batch_size=1,
    device=None
):
    frame_lists, initial_poses, next_motions, labels, object_boxes = load_data(dataset_path)
    # print("initial_poses shape:", len(initial_poses))
    # print("object_boxes shape:", len(object_boxes))
    # print("next_motion_descriptions shape:", len(next_motion_descriptions))
    # print("next_motions shape:", len(next_motions))

    # print("initial_poses shape:", initial_poses[0].shape)
    # print("object_boxes shape:", object_boxes[0].shape)
    # print("next_motion_descriptions shape:", next_motion_descriptions[0].shape)
    # print("next_motions shape:", next_motions[0].shape)

    train_dataset, val_dataset, test_dataset = create_datasets(frame_lists, initial_poses, next_motions, labels, object_boxes, device)

    # create dataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
