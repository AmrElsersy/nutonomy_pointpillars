# -*- coding: utf-8 -*-
from cProfile import label
from distutils.log import info
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
from second.core import box_np_ops

# ==========================
sys.path.append('../../')

from second.core import box_np_ops
import cv2
from visualization.KittiUtils import BBox2D, BBox3D, KittiObject, KittiCalibration
from visualization.KittiVisualization import KittiVisualizer
from visualization.KittiDataset import KittiDataset
visualizer = KittiVisualizer()
score_threshold = 0.2

import gc
gc.collect()
torch.cuda.empty_cache()
def pointpillars_output_to_kitti_objects(predictions):
    predictions = predictions[0]
    n = len(predictions['name'])
    # print(len(predictions['score']), predictions['location'])

    kitti_objects = []
    for i in range(n):
        bbox = predictions['bbox'][i]
        dims = predictions['dimensions'][i]
        location = predictions['location'][i]
        rotation = predictions['rotation_y'][i]

        # z coord is center in one coordinate and bottom in the other
        location[2] -= location[2]/2

        score = predictions['score'][i]
        if score < score_threshold:
            continue

        bbox = BBox2D(bbox) # 0 1 2, 2 0 1, ... 1 0 2, 1 2 0
        box3d = BBox3D(location[0], location[1], location[2], dims[1], dims[2], dims[0], -rotation)
        kitti_object = KittiObject(box3d, 1, score, bbox)

        kitti_objects.append(kitti_object)
    return kitti_objects

def visualize(pointcloud, predictions, image=None, calib=None):
    global visualizer
    # predictions = pointpillars_output_to_kitti_objects(predictions)

    if image is None:
        visualizer.visualize_scene_bev(pointcloud=pointcloud, objects=predictions)
    else:
        visualizer.visualize_scene_2D(pointcloud, image, predictions, calib=calib)

    if visualizer.user_press == 27:
        cv2.destroyAllWindows()
        exit()

def gt_boxes_to_kitti_objects(gt_boxes, rect, Trv2c):
    # convert boxes to lidar coordinates
    gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, rect, Trv2c)
    objects = []
    for box in gt_boxes:
        location = box[0:3]
        dims = box[3:6]
        rotation = box[6]

        box3d = BBox3D(location[0], location[1], location[2], dims[1], dims[2], dims[0], -rotation)
        kitti_object = KittiObject(box3d, 1)
        objects.append(kitti_object)

    return objects

def test():
    config_path = 'configs/pointpillars/car/xyres_16.proto'
    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    #########################
    # Build Voxel Generator
    #########################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    dataset.dataset.return_input=True

    normal_dataset = KittiDataset(input_cfg.kitti_root_path + "training")

    for i in range(len(dataset)):
        example = dataset[i]
        infos = dataset.dataset.kitti_infos[i]

        pointcloud_processed = example['pointcloud']
        
        path_pointcloud = infos['velodyne_path']
        path_image = infos['img_path']
        
        rect = example['rect']
        Trv2c = example['Trv2c']
        gt_boxes = example['gt_boxes']
        
        labels = gt_boxes_to_kitti_objects(gt_boxes, rect, Trv2c)
        visualize(pointcloud_processed, labels)
        for label in labels:
            print('pos= ', label.bbox_3d.pos, '.. dims= ', label.bbox_3d.dims)

        image_idx = example['image_idx']
        image, pointcloud, labels2, calib = normal_dataset[image_idx]
        visualizer.visualize_scene_2D(pointcloud, image, [], labels=labels2, calib=calib)
        print("Len of normal ", len(labels2), ' .. ', len(labels))
        print("===================")

        # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
        #               4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
        #               8: 'image_idx', 9: 'image_shape']

if __name__ == '__main__':
    test()
