# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from mdetr.datasets.mixed import CustomCocoDetection
from mdetr.datasets.clevr import build as build_clevr
from mdetr.datasets.clevrref import build as build_clevrref
from mdetr.datasets.coco import build as build_coco
from mdetr.datasets.flickr import build as build_flickr
from mdetr.datasets.gqa import build as build_gqa
from mdetr.datasets.lvis import LvisDetectionBase
from mdetr.datasets.lvis import build as build_lvis
from mdetr.datasets.lvis_modulation import build as build_modulated_lvis
from mdetr.datasets.mixed import build as build_mixed
from mdetr.datasets.phrasecut import build as build_phrasecut
from mdetr.datasets.refexp import build as build_refexp



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, LvisDetectionBase):
        return dataset.lvis
    if isinstance(dataset, (torchvision.datasets.CocoDetection, CustomCocoDetection)):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if "clevrref" in dataset_file:
        return build_clevrref(image_set, args)
    if "clevr" in dataset_file:
        return build_clevr(dataset_file, image_set, args)
    if dataset_file == "coco":
        return build_coco(image_set, args)
    if dataset_file == "flickr":
        return build_flickr(image_set, args)
    if dataset_file == "gqa":
        return build_gqa(image_set, args)
    if dataset_file == "lvis":
        return build_lvis(image_set, args)
    if dataset_file == "modulated_lvis":
        return build_modulated_lvis(image_set, args)
    if dataset_file == "mixed":
        return build_mixed(image_set, args)
    if dataset_file == "refexp":
        return build_refexp(image_set, args)
    if dataset_file == "vg":
        return build_vg(image_set, args)
    if dataset_file == "phrasecut":
        return build_phrasecut(image_set, args)
    raise ValueError(f"dataset {dataset_file} not supported")
