import json
import numpy
import os
import requests
import torch
import torch.nn as nn

from requests.exceptions import RequestException
from tqdm import tqdm


GPT_2_SMALL = '124M'
GPT_2_MEDIUM = '355M'
GPT_2_LARGE = '774M'
GPT_2_XL = '1558M'


def model_directory(model_size: str):
    return os.path.join('../resources/model-weights', model_size)

def download_weights(model_size: str):
    allowed_sizes = (GPT_2_SMALL, GPT_2_MEDIUM, GPT_2_LARGE, GPT_2_XL)
    if model_size not in allowed_sizes:
        raise ValueError(f'Invalid model size: {model_size}')

    model_dir = model_directory(model_size)
    base_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models'
    filenames = [
        'checkpoint', 'encoder.json', 'hparams.json',
        'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
        'model.ckpt.meta', 'vocab.bpe'
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)


def download_file(url: str, destination: str):
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get('Content-Length', 0))

        if os.path.exists(destination):
            local_file_size = os.path.getsize(destination)
            if file_size == local_file_size:
                print(f'File already exists and is up-to-date: {destination}')
                return True

        block_size = 1024
        desc = os.path.basename(url)
        with tqdm(total=file_size, unit='iB', unit_scale=True, desc=desc) as progress_bar:
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True
    except RequestException as ex:
        print(f'Error while downloading from {url}: {ex}')


def load_weights(model_size: str) -> tuple[dict, dict]:
    import tensorflow
    model_dir = model_directory(model_size)
    tf_ckpt_path = tensorflow.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    print('Settings: ', settings)
    print('Parameter dictionary keys: ', params.keys())
    return settings, params


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    import tensorflow
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tensorflow.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = numpy.squeeze(tensorflow.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f'Shape mismatch: {left.shape} != {right.shape}')

    return nn.Parameter(torch.tensor(right))

def assign_weights(gpt, params):
    # Embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # Transformer blocks
    for b in range(len(params['blocks'])):
        q_w, k_w, v_w = numpy.split((params['blocks'][b]['attn']['c_attn'])['w'], 3, axis=-1)
        gpt.trf_blocks[b].attention.W_query.weight = assign(gpt.trf_blocks[b].attention.W_query.weight, q_w.T)
        gpt.trf_blocks[b].attention.W_key.weight = assign(gpt.trf_blocks[b].attention.W_key.weight, k_w.T)
        gpt.trf_blocks[b].attention.W_value.weight = assign(gpt.trf_blocks[b].attention.W_value.weight, v_w.T)

        # Q, K, V bias
        q_b, k_b, v_b = numpy.split((params['blocks'][b]['attn']['c_attn'])['b'], 3, axis=-1)
        gpt.trf_blocks[b].attention.W_query.bias = assign(gpt.trf_blocks[b].attention.W_query.bias, q_b)
        gpt.trf_blocks[b].attention.W_key.bias = assign(gpt.trf_blocks[b].attention.W_key.bias, k_b)
        gpt.trf_blocks[b].attention.W_value.bias = assign(gpt.trf_blocks[b].attention.W_value.bias, v_b)

        gpt.trf_blocks[b].attention.out_proj.weight = assign(
            gpt.trf_blocks[b].attention.out_proj.weight,
            params['blocks'][b]['attn']['c_proj']['w'].T
        )
        gpt.trf_blocks[b].attention.out_proj.bias = assign(
            gpt.trf_blocks[b].attention.out_proj.bias,
            params['blocks'][b]['attn']['c_proj']['b']
        )

        gpt.trf_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.trf_blocks[b].feed_forward.layers[0].weight,
            params['blocks'][b]['mlp']['c_fc']['w'].T
        )
        gpt.trf_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.trf_blocks[b].feed_forward.layers[0].bias,
            params['blocks'][b]['mlp']['c_fc']['b']
        )
        gpt.trf_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.trf_blocks[b].feed_forward.layers[2].weight,
            params['blocks'][b]['mlp']['c_proj']['w'].T
        )
        gpt.trf_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.trf_blocks[b].feed_forward.layers[2].bias,
            params['blocks'][b]['mlp']['c_proj']['b']
        )

        gpt.trf_blocks[b].layer_norm_1.scale = assign(
            gpt.trf_blocks[b].layer_norm_1.scale,
            params['blocks'][b]['ln_1']['g']
        )
        gpt.trf_blocks[b].layer_norm_1.shift = assign(
            gpt.trf_blocks[b].layer_norm_1.shift,
            params['blocks'][b]['ln_1']['b']
        )
        gpt.trf_blocks[b].layer_norm_2.scale = assign(
            gpt.trf_blocks[b].layer_norm_2.scale,
            params['blocks'][b]['ln_2']['g']
        )
        gpt.trf_blocks[b].layer_norm_2.shift = assign(
            gpt.trf_blocks[b].layer_norm_2.shift,
            params['blocks'][b]['ln_2']['b']
        )
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params['g'])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params['b'])
    gpt.output_layer.weight = assign(gpt.output_layer.weight, params['wte'])


def pretrain(model: nn.Module):
    if model.state_file.exists():
        model.load()
    else:
        if not os.path.exists(f'../resources/model-weights/{GPT_2_SMALL}'):
            download_weights(GPT_2_SMALL)

        settings, params = load_weights(GPT_2_SMALL)
        assign_weights(model, params)
        model.save()
