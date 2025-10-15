import base64
import json
import os
import requests
import yaml
from typing import Any


def load_encode_image(img_path: str):
    """
    Load and encode image as a base64 string.

    Parameters
    ----------
    img_path : str
        Path to input image.

    Returns
    -------
    img_b64 : str
        Input encoded and decoded as base64 bytes string.
    """

    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_b64


def post(params: dict[str, Any], host="127.0.0.1", port=8080):
    """
    Send POST request with config params to API Server.

    Parameters
    ----------
    params : dict[str, Any]
        Dict of configuration parameters for generation request.
    host : str, optional, default="127.0.0.1"
        IP address for hosting API Server.
    port : int, optional, default=8080
        Port for API Server.

    Returns
    -------
    response : Response
        HTTP response.
    """

    response = requests.post(
        f"http://{host}:{port}/generate",
        headers={'Content-Type': "application/json"},
        data=json.dumps(params),
    )

    return response


def output_result(response: requests.Response, output_dir="results", output_file_type='glb'):
    """
    Save result to file.

    Parameters
    ----------
    response : Response
        HTTP response.
    output_dir : str, optional, default="results"
        Path to output directory.
    output_file_type : str, optional, default='glb'
        File format for output mesh.

    Returns
    -------
    output_path : str
        Path to output mesh file.
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{response.headers['uid']}.{output_file_type}"
    output_path = os.path.join(output_dir, file_name)

    # Write mesh from HTTP response to file
    with open(output_path, 'wb') as f:
        f.write(response.content)

    return output_path


def pipeline(image: list[str] = None, output_dir="results", config_path="config.yaml", **kwargs):
    """
    Execute POST request pipeline.

    Parameters
    ----------
    image : list[str], optional, default=None
        input image/s.
    output_dir : str, optional, default="results"
        Path to output file directory.
    config_path : str, optional, default="config.yaml"
        Path to configuration YAML file.

    Returns
    -------
    response : Response
        HTTP response.
    output_path : str
        Path to output mesh file.

    Raises
    ------
    ValueError
        No image provided.
    """

    # 1. Load configs
    with open(config_path, 'rb') as f:
        config = yaml.safe_load(f)

    # 2. Load and encode each image as a base64 string
    if image is not None:
        image_b64 = [load_encode_image(img) for img in image]
        config['inference']['image'] = image_b64
    else:
        raise ValueError("No image provided.")

    # 3. Send HTTP POST request
    response = post(config, **kwargs)

    # 4. Save result response to specified(?) file
    output_path = output_result(response, output_dir) if response.ok else None

    return response, output_path


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, action='extend', nargs='+', required=True, help="Path/s to input image file/s")
    parser.add_argument('--output-dir', '-o', type=str, default="results", help="Path to output directory")
    parser.add_argument('--config-path', '-c', type=str, default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument('--host', type=str, default="127.0.0.1", help="API Server host IP address")
    parser.add_argument('--port', '-p', type=int, default=8080, help="API Server port")
    args = parser.parse_args()

    config_dir = "config"
    args.config_path = os.path.join(config_dir, args.config_path)

    response, output_path = pipeline(**vars(args))