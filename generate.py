import os
import time
from typing import Any

import torch
import uuid
import yaml
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline


def generate_from_img(image: list[str], output_dir: str, config: dict[str, Any] | str):
    """
    Inference script to generate model from input image.

    Parameters
    ----------
    image : list[Any]
        Path/s to input image/s.
    output_dir : str
        Path to output file directory.
    config : dict[str, Any] | str
        Configuration dict or path to configuration YAML file.

    Returns
    -------
    mesh : Trimesh
        Generated 3D model mesh.
    """

    # 1. Load params
    if isinstance(config, str):
        with open(config, 'rb') as f:
            params = yaml.safe_load(f)
    else:
        params = config

    if params['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available. Using CPU as device instead.")
        params['device'] = 'cpu'

    uid = uuid.UUID()
    ext = params['output_file_type']

    # 2. Load image
    image = [Image.open(img).convert('RGBA') for img in image]

    # 3. Remove backgroud from image
    rembg = BackgroundRemover(params['rembg']['model_name'])
    image = [rembg(img, **params['rembg']) for img in image]
    #img.save(os.path.join(output_dir, f"{obj_name}_rembg.png"))
    params['inference']['image'] = image[0]

    ######### CONTINUE CHANGING IMAGE INPUTS TO LISTS #########

    # 4. Model pipelines
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(**params['model'], device=params['device'])

    # 5. Generate model
    t0 = time.time()
    mesh = pipeline(**params['inference'])[0]
    t1 = time.time()
    print(f"--- Model Shape Generation: {t1 - t0:.3f} secs ---")

    # 6. Perform mesh post-processing
    mesh = FloaterRemover()(mesh, params['nbfaceratio'])
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, params['max_facenum'])

    # 7. Output model mesh
    mesh.export(os.path.join(output_dir, f"{uid}.{ext}"))

    # 8. Empty cuda cache
    torch.cuda.empty_cache()

    return mesh


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, action='extend', nargs='+', required=True, help="Path to input image file")
    parser.add_argument('--output-dir', '-o', type=str, default="results", help="Path to output directory")
    parser.add_argument('--config-path', '-c', type=str, default="config.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()

    config_dir = "config"
    args.config_path = os.path.join(config_dir, args.config_path)

    generate_from_img(**vars(args))