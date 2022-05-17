# -*- coding: utf-8 -*-
import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import sys
from xml.dom import expatbuilder
sys.path.append('../')
from pathlib import Path
import fire
import numpy as np
import torch
from google.protobuf import text_format
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.train import get_paddings_indicator, example_convert_to_torch, predict_kitti_to_anno

# ==========================
sys.path.append('../../')

from second.core import box_np_ops
import cv2
from visualization.KittiUtils import BBox2D, BBox3D, KittiObject, KittiCalibration
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiDataset import KittiDataset
visualizer = KittiVisualizer()
score_threshold = 0.5

import gc
gc.collect()
torch.cuda.empty_cache()
def pointpillars_output_to_kitti_objects(predictions):
    predictions = predictions[0]
    n = len(predictions['name'])
    # print(len(predictions['score']), predictions['location'])
    # print(predictions.keys())

    classes_names = {
        "Car": 0,
        "Truck": 1,
        "Cyclist": 2
    }

    classes_score_threshold = {
        "Car": 0.5,
        "Truck": 0.4,
        "Cyclist": 0.25
    }

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

def visualize(pointcloud, predictions, image=None, calib=None, labels=[]):
    global visualizer

    predictions = pointpillars_output_to_kitti_objects(predictions)

    if image is None:
        visualizer.visualize_scene_bev(pointcloud=pointcloud, objects=predictions)
    else:
        visualizer.visualize_scene_2D(pointcloud, image, predictions, labels=labels, calib=calib)

    if visualizer.user_press == 27:
        cv2.destroyAllWindows()
        exit()

def test(config_path='configs/pointpillars/car/xyres_16.proto',
         model_dir='/path/to/model_dir',
        #  dataset_path='/home/kitti_original/training',
         dataset_path='/home/kitti_original/testing',
        #  dataset_path='/home/kitti/dataset/kitti/training',
        #  dataset_path='/home/kitti/dataset/kitti/testing',
        #  checkpoint='/home/nutonomy_pointpillars/voxelnet-44649.tckpt'
        checkpoint=None
        ):

    model_dir = str(Path(model_dir).resolve())

    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

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


        visualize(pointcloud, predictions, image, calib, labels)

if __name__ == '__main__':
    fire.Fire()
