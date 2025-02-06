# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Dinov2 checkpoints trained with the DINO method."""

import argparse
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import Dinov2Config, ViTImageProcessor, Dinov2Model
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"dinov2.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"dinov2.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"dinov2.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"dinov2.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"dinov2.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"dinov2.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"dinov2.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"dinov2.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"dinov2.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"dinov2.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "dinov2.embeddings.cls_token"),
            ("patch_embed.proj.weight", "dinov2.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "dinov2.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "dinov2.embeddings.position_embeddings"),
        ]
    )

    # layernorm + pooler
    rename_keys.extend(
        [
            ("norm.weight", "layernorm.weight"),
            ("norm.bias", "layernorm.bias"),
        ]
    )

    # if just the base model, we should remove "dinov2" from all keys that start with "dinov2"
    rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("dinov2") else pair for pair in rename_keys]
    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        prefix = ""
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

def get_dinov2_config(model_name):
    config = Dinov2Config(image_size=518, patch_size=14)
    if "vits" in model_name:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    elif "vitb" in model_name:
        raise NotImplementedError("To Do")
    elif "vitl" in model_name:
        raise NotImplementedError("To Do")
    elif "vitg" in model_name:
        raise NotImplementedError("To Do")
    else:
        raise ValueError("Model not supported")
    
    return config

@torch.no_grad()
def convert_dinov2_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our Dinov2 structure.
    """

    # define default Dinov2 configuration
    config = get_dinov2_config()

    # load original model from torch hub
    original_model = torch.hub.load("facebookresearch/dino:main", model_name)
    original_model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = original_model.state_dict()
    remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    model = Dinov2Model(config, add_pooling_layer=False).eval()

    # Check outputs on an image, prepared by ViTImageProcessor
    image_processor = ViTImageProcessor()
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)


    final_hidden_state_cls_token = original_model(pixel_values)
    assert torch.allclose(final_hidden_state_cls_token, outputs.last_hidden_state[:, 0, :], atol=1e-1)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dinov2_vitb14",
        type=str,
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    args = parser.parse_args()
    convert_dinov2_checkpoint(args.model_name, args.pytorch_dump_folder_path)
