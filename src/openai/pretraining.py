import json
import numpy
import os
import requests
import tensorflow

from requests.exceptions import RequestException
from tqdm import tqdm


def model_directory(model_size: str):
    return os.path.join('resources/model-weights', model_size)


def download_weights(model_size: str):
    allowed_sizes = ('124M', '355M', '774M', '1558M')
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


def load_weights(model_size: str):
    model_dir = model_directory(model_size)
    tf_ckpt_path = tensorflow.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
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