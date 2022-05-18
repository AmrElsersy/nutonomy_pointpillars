# -*- coding: utf-8 -*-
import enum
import time
import sys, os
sys.path.append('../')
from pathlib import Path
import fire
import numpy as np
import torch
from google.protobuf import text_format
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, second_builder)
from second.pytorch.train import example_convert_to_torch, predict_kitti_to_anno

# ==========================
sys.path.append('../../')
from second.core import box_np_ops
import cv2
from visualization.KittiUtils import BBox2D, BBox3D, KittiObject, KittiCalibration, label_to_class_name
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiDataset import KittiDataset

visualizer = KittiVisualizer()
torch.cuda.empty_cache()

classes_names = {
    "Car": 0,
    "Truck": 1,
    "Cyclist": 2
}

classes_score_threshold = {
    "Car": 0.5,
    "Truck": 0.3,
    "Cyclist": 0.2
}

def pointpillars_output_to_kitti_objects(predictions):
    predictions = predictions[0]
    n = len(predictions['name'])


    kitti_objects = []
    for i in range(n):
        bbox = predictions['bbox'][i]
        dims = predictions['dimensions'][i]
        location = predictions['location'][i]
        rotation = predictions['rotation_y'][i]
        label_class = predictions["name"][i]
        score = predictions['score'][i]

        # z coord is center in one coordinate and bottom in the other
        location[2] -= location[2]/2

        if score < classes_score_threshold[label_class]:
            # print("skipped ", label_class, ' with score=',score)
            continue

        bbox = BBox2D(bbox) # 0 1 2, 2 0 1, ... 1 0 2, 1 2 0
        box3d = BBox3D(location[0], location[1], location[2], dims[1], dims[2], dims[0], -rotation)
        kitti_object = KittiObject(box3d, classes_names[label_class], score, bbox)

        kitti_objects.append(kitti_object)
    return kitti_objects

def visualize(pointcloud, predictions, image=None, calib=None, labels=[], print_detections=False):
    global visualizer

    predictions = pointpillars_output_to_kitti_objects(predictions)

    if(print_detections):
        num_predictions = [0, 0, 0]
        for det in predictions:
            num_predictions[det.label] += 1
        
        label_to_name = {}
        for key in classes_names:
            label_to_name[classes_names[key]] = key

        print("Detected: ")
        for i, n_pred in enumerate(num_predictions):
            print(n_pred, " ", label_to_name[i])

    if image is None:
        visualizer.visualize_scene_bev(pointcloud=pointcloud, objects=predictions)
    else:
        visualizer.visualize_scene_2D(pointcloud, image, predictions, labels=labels, calib=calib)
        # visualizer.visualize_scene_3D(pointcloud=pointcloud, objects=predictions, calib=calib)

    if visualizer.user_press == 27:
        cv2.destroyAllWindows()
        exit()

def test(config_path='configs/pointpillars/car/xyres_16.proto',
         model_dir='/path/to/model_dir',
        #  checkpoint='/home/nutonomy_pointpillars/voxelnet-44649.tckpt'
        checkpoint=None,
        mode="testing" # can be training or testing to open the training/testing folder in the dataset specified in the config
        ):

    model_dir = str(Path(model_dir).resolve())

    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    dataset_path = str(config.train_input_reader.kitti_root_path)
    dataset_path = os.path.join(dataset_path, mode)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    #########################
    # Build Voxel Generator
    #########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    net = second_builder.build(model_cfg, voxel_generator, target_assigner, 1)
    net.cuda()

    if checkpoint is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(checkpoint, net)

    float_dtype = torch.float32

    net.eval()

    dataset = KittiDataset(dataset_path)

    for i in range(len(dataset)):
        image, pointcloud, labels, calib = dataset[i]
        pointcloud = pointcloud.reshape(-1,4)

        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = voxel_generator.voxel_size # pillar size
        pc_range = voxel_generator.point_cloud_range # clip pointcloud ranges
        grid_size = voxel_generator.grid_size # ground size 
        max_voxels = 20000
        anchor_area_threshold = 1 
        out_size_factor =2 # rpn first downsample stride / rpn first upsample stride

        feature_map_size = grid_size[:2] // out_size_factor
        feature_map_size = [*feature_map_size, 1][::-1]
        # [352, 400]

        voxels, coordinates, num_points = voxel_generator.generate(
            pointcloud, max_voxels)

        # Anchors from target assigner
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold

        example = {
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            'rect': np.identity(4, dtype=np.float32),
            'Trv2c': np.identity(4, dtype=np.float32),
            'P2': np.identity(4, dtype=np.float32),
            "anchors": anchors,
            'anchors_mask': anchors_mask,
            'image_idx': torch.tensor([i]),
            'image_shape': torch.tensor([1242, 35]).reshape((1,2))
        }

        #  [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
        #  4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
        #  8: 'image_idx', 9: 'image_shape']

        # print('voxels ',example['voxels'].shape)
        # print('num_points ',example['num_points'].shape)
        # print('coordinates ',example['coordinates'].shape)
        # print('anchors ',example['anchors'].shape)
        # print('anchors_mask ',example['anchors_mask'].shape)

        coordinates = np.pad(
                coordinates, 
                ((0, 0), (1, 0)),
                mode='constant',
                constant_values=1)
        example['coordinates'] = torch.from_numpy(coordinates)

        example = example_convert_to_torch(example, float_dtype)

        example['voxels'] = torch.squeeze(example['voxels'], 0)
        example['num_points'] = torch.squeeze(example['num_points'], 0)
        example['anchors'] = example['anchors'].unsqueeze(0)
        example['anchors_mask'] = example['anchors_mask'].unsqueeze(0)
        example['Trv2c'] = example['Trv2c'].unsqueeze(0)
        example['rect'] = example['rect'].unsqueeze(0)
        example['P2'] = example['P2'].unsqueeze(0)

        example_tuple = list(example.values())

        # for key in example:
        #     if torch.is_tensor(example[key]) or type(example[key] )==np.ndarray:
        #         print(key,example[key].shape)
        # print("#############")
        # for i, ex in enumerate(example_tuple):
        #     if torch.is_tensor(ex) or type(ex)==np.ndarray:
        #         print(i, ex.shape)
        #     else :
        #         print(i, " = ", ex)
        # print("####################################################")
     
        with torch.no_grad():
            predictions = predict_kitti_to_anno(
                net, example_tuple, class_names, center_limit_range,
                model_cfg.lidar_input, None)


        visualize(pointcloud, predictions, image, calib, labels, print_detections=True)

if __name__ == '__main__':
    fire.Fire()
