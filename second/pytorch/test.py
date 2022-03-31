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

import cv2
from visualization.KittiUtils import BBox2D, BBox3D, KittiObject, KittiCalibration
from visualization.KittiVisualization import KittiVisualizer
visualizer = KittiVisualizer()
score_threshold = 0.2

import gc
gc.collect()
torch.cuda.empty_cache()
def pointpillars_output_to_kitti_objects(predictions, calib):
    predictions = predictions[0]
    n = len(predictions['name'])
    print(len(predictions['score']), predictions['location'])

    kitti_objects = []
    for i in range(n):
        bbox = predictions['bbox'][i]
        dims = predictions['dimensions'][i]
        location = predictions['location'][i]
        rotation = predictions['rotation_y'][i]

        # convert center from 3d camera coord to velodyne coord
        print('location before ', location, location.shape)
        location = calib.rectified_camera_to_velodyne(location.reshape(1,3))
        print('location after ', location, location.shape)

        location = location.reshape(4,)[:3]

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

def visualize(example, predictions):
    global visualizer

    image_path = os.path.join(os.environ["KITTI_DATASET_ROOT"], example['image_path'][0])
    image = cv2.imread(image_path)
    pointcloud = example['pointcloud'].squeeze(0).numpy()

    Trv2c = example['Trv2c'].detach().cpu().squeeze(0).numpy()
    P2 = example['P2'].detach().cpu().squeeze(0).numpy()
    rect = example['rect'].detach().cpu().squeeze(0).numpy()

    Trv2c = Trv2c[:3, :]
    P2 = P2[:3, :]
    rect = rect[:3, :3]

    calib = KittiCalibration(Trv2c, P2, rect)
    predictions = pointpillars_output_to_kitti_objects(predictions, calib)

    v = visualizer.visualize_scene_2D(pointcloud=pointcloud, 
        image=image, objects=predictions, calib=calib, visualize=False)
    cv2.imshow("output", v)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        exit()

def test(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):

    model_dir = str(Path(model_dir).resolve())
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)

    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
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
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        return_input=True)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=input_cfg.num_workers,
        pin_memory=False
        )

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()

    for example in iter(eval_dataloader):
        # eval example [0: 'voxels', 1: 'num_points', 2: 'coordinates', 3: 'rect'
        #               4: 'Trv2c', 5: 'P2', 6: 'anchors', 7: 'anchors_mask'
        #               8: 'image_idx', 9: 'image_shape']

        coord_numpy = example['coordinates'].numpy()
        coord_numpy = coord_numpy.squeeze(0)
        coord_numpy = np.pad(
                coord_numpy, 
                ((0, 0), (1, 0)),
                mode='constant',
                constant_values=1)
        example['coordinates'] = torch.from_numpy(coord_numpy)

        example.pop("num_voxels")

        example = example_convert_to_torch(example, float_dtype)

        example['voxels'] = torch.squeeze(example['voxels'], 0)
        example['num_points'] = torch.squeeze(example['num_points'], 0)

        example_tuple = list(example.values())

        for key in example:
            if torch.is_tensor(example[key]) or type(example[key] )==np.ndarray:
                print(key,example[key].shape)
        print("#############")
        for i, ex in enumerate(example_tuple):
            if torch.is_tensor(ex) or type(ex)==np.ndarray:
                print(i, ex.shape)
            else :
                print(i, " = ", ex)
        print("####################################################")
     
        with torch.no_grad():
            predictions = predict_kitti_to_anno(
                net, example_tuple, class_names, center_limit_range,
                model_cfg.lidar_input, None)

        visualize(example, predictions)

if __name__ == '__main__':
    fire.Fire()
