# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import logging
import os
import re
import sys
from functools import wraps

import torch


def get_logger(name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = get_logger('hy3dgen.shapegen')


class synchronize_timer:
    """
    Synchronized timer to count the inference time of `nn.Module.forward`. Supports both context manager and decorator usage.

    Example as context manager:
    ```python
    with synchronize_timer('name') as t:
        run()
    ```

    Example as decorator:
    ```python
    @synchronize_timer("Export to trimesh")
    def export_to_trimesh(mesh_output):
        pass
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f"{self.name} takes {self.time} ms")

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result
        return wrapper


def smart_load_model(model_path, subfolder, use_safetensors, variant):
    original_model_path = model_path

    # Try local path
    base_dir = re.sub(r"(?<=hy3dgen\W).+", "models", os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, *os.path.split(model_path))
    model_path = os.path.join(model_dir, subfolder)

    ext = 'safetensors' if use_safetensors else 'ckpt'
    variant = '' if variant is None else f'.{variant}'
    config_name, ckpt_name = "config.yaml", f"model{variant}.{ext}"

    logger.info(f"Trying to load model from local path: {os.path.relpath(model_path)}")
    if not os.path.exists(os.path.join(model_path, ckpt_name)):
        logger.info("Model path does not exist, trying to download from huggingface.")
        try:
            from huggingface_hub import snapshot_download
            # Download only specified subdirectory (只下载指定子目录)
            # Key modification: Pattern matching subfolders (关键修改：模式匹配子文件夹)
            os.makedirs(base_dir, exist_ok=True)
            path = snapshot_download(
                repo_id=original_model_path, local_dir=model_dir,
                allow_patterns=[os.path.join(subfolder, f) for f in [config_name, ckpt_name]],
            )
            # Keep path splicing logic unchanged (保持路径拼接逻辑不变)
            model_path = os.path.join(path, subfolder)
        except ImportError:
            logger.warning("You need to install HuggingFace Hub to load models from the hub.")
            raise RuntimeError(f"Model path {model_path} not found.")
        except Exception as e:
            raise e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {original_model_path} not found.")

    config_path, ckpt_path = [os.path.join(model_path, f) for f in [config_name, ckpt_name]]

    return config_path, ckpt_path
