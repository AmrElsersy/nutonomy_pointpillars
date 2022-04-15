# -*- coding: utf-8 -*-
from cProfile import label
from distutils.log import info
from genericpath import exists
import sys
from xml.dom import expatbuilder
sys.path.append('../')
from pathlib import Path
import numpy as np
from google.protobuf import text_format
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
        pointcloud_processed = example['pointcloud']        
        rect = example['rect']
        Trv2c = example['Trv2c']
        gt_boxes = example['gt_boxes']
        
        # labels from kitti infos
        labels = gt_boxes_to_kitti_objects(gt_boxes, rect, Trv2c)
        # visualization from pointpillars repo labels & pointcloud
        visualizer.visualize_scene_bev(pointcloud=pointcloud_processed, objects=labels)

        # visualization from my kitti_dataset labels & pointcloud
        image_idx = example['image_idx']
        image, pointcloud, labels2, calib = normal_dataset[image_idx]
        visualizer.visualize_scene_2D(pointcloud, image, [], labels=labels2, calib=calib)

        if visualizer.user_press == 27:
            cv2.destroyAllWindows()
            exit()

        print("My Dataset ", len(labels2), ' .. Repo dataset ', len(labels))
        print("===================")

if __name__ == '__main__':
    test()
