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

import asyncio
import base64
import logging
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO
from typing import Any

import torch
import torch.types
import trimesh
import uvicorn
import yaml
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
handler = None


def build_logger(logger_name: str, logger_filename: str):
    """
    Build logger and redirect writes to its instance.

    Parameters
    ----------
    logger_name : str
        Name of logger.
    logger_filename : str
        Name of logger file.

    Returns
    -------
    logger : Logger
        Built logger.
    """

    global handler
    LOGDIR = "."

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for _, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance."""

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            # > On output, if newline is None, any '\n' characters written are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


#def pretty_print_semaphore(semaphore):
#    if semaphore is None:
#        return "None"
#    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = "output_cache"
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", os.path.join(SAVE_DIR, "controller.log"))


def load_image_from_base64(image: str):
    """
    Load and decode base64 image.

    Parameters
    ----------
    image : str
        Input image base64 string.

    Returns
    -------
    image : Image
        Input image decoded as RGBA PIL Image.
    """

    return Image.open(BytesIO(base64.b64decode(image))).convert('RGBA')


class ModelWorker:

    def __init__(
        self, model_path="tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", use_safetensors=True, device: torch.types.Device = 'cuda',
        rembg_model="isnet-general-use", **kwargs,
    ):
        """
        A model worker executes the model.

        Parameters
        ----------
        model_path : str, optional, default="tencent/Hunyuan3D-2mini"
            Name of pretrained model HuggingFace repository.
        subfolder : str, optional, default="hunyuan3d-dit-v2-mini"
            Name of specific model subfolder in pretrained model Huggingface repository.
        use_safetensors : bool, optional, default=True
            Whether to only use safetensors files for pretained models.
        rembg_model : str, optional, default="isnet-general-use"
            Name of rembg model to use for image background removal.
        device : Device, optional, default='cuda'
            Device to use for Pytorch and model generation.
        """

        self.model_path = model_path
        self.subfolder = subfolder
        self.use_safetensors = use_safetensors
        self.worker_id = worker_id
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available. Using CPU as device instead.")
            self.device = 'cpu'
        else:
            self.device = device
        logger.info(f"Loading model {model_path} on worker {worker_id}...")

        self.rembg = BackgroundRemover(rembg_model)
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.model_path, subfolder=self.subfolder, use_safetensors=self.use_safetensors, device=self.device,
        )
        #self.pipeline.enable_flashvdm(mc_algo='mc')

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {'speed': 1, 'queue_length': self.get_queue_length()}

    @torch.inference_mode()
    def generate(self, uid: uuid.UUID, params: dict[str, Any]):
        """
        Generate 3D model from HTTP request.

        Parameters
        ----------
        uid : uuid.UUID
            Unique job ID.
        params : dict[str, Any]
            Generation and model configuration parameters.

        Returns
        -------
        sve_path : _type_
            _description_.
        uid : UUID
            Universally Unique Identifier for generation job request.

        Raises
        ------
        ValueError
            No input image provided.
        """

        if 'image' in params['inference']:
            image = [load_image_from_base64(img) for img in params['inference']['image']]
        else:
            raise ValueError("No input image provided.")

        image = [self.rembg(img, **params['rembg']) for img in image]
        params['inference']['image'] = image[0]

        import time
        t0 = time.time()
        mesh = self.pipeline(**params['inference'])[0]
        t1 = time.time()
        logger.info("--- %s seconds ---" % (t1 - t0))

        mesh = FloaterRemover()(mesh, nbfaceratio=params['nbfaceratio'])
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=params['max_facenum'])

        file_type = params['output_file_type']
        with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f"{str(uid)}.{file_type}")
            mesh.export(save_path)
            if os.path.exists(save_path):
                logger.info(f"Exported mesh saved to {save_path}")

        torch.cuda.empty_cache()

        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # You can specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # Allow all methods
    allow_headers=["*"],    # Allow all headers
)


@app.post("/generate")
async def generate(request: Request):
    """
    Execute generation POST request (asynchronous).

    Parameters
    ----------
    request : Request
        _description_.

    Returns
    -------
    response : FileResponse | JSONResponse
        HTTP response containing either:
        - `FileResponse` : Output file path to generated model and associated UID.
        - `JSONResponse` : `server_error_msg` and status code.
    """

    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()

    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path, headers={'uid': str(uid)})
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        ret = {'text': server_error_msg, 'error_code': 1}
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        ret = {'text': server_error_msg, 'error_code': 1}
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        traceback.print_exc()
        print("Caught Unknown Error:", e)
        ret = {'text': server_error_msg, 'error_code': 1}
        return JSONResponse(ret, status_code=404)


@app.post("/send")
async def generate(request: Request):
    """
    Send generation job request (?) (asynchronous).

    Parameters
    ----------
    request : Request
        HTTP request.

    Returns
    -------
    response : JSONResponse
        HTTP response containing UID and status code.
    """

    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()

    threading.Thread(target=worker.generate, args=(uid, params)).start()
    ret = {'uid': str(uid)}

    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    """
    Request generation job status (asynchronous).

    Parameters
    ----------
    uid : str
        Generation job request UID.

    Returns
    -------
    response : JSONResponse
        HTTP response containing job status and status code.
    """

    save_file_path = os.path.join(SAVE_DIR, f"{uid}.glb")

    if not os.path.exists(save_file_path):
        response = {'status': "processing"}
    else:
        with open(save_file_path, 'rb') as f:
            base64_str = base64.b64encode(f.read()).decode()
            response = {'status': "completed", 'model_base64': base64_str}

    return JSONResponse(response, status_code=200)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default="0.0.0.0", help="API Server host IP address")
    parser.add_argument('--port', '-p', type=int, default=8080, help="API Server port")
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'cuda'], default='cuda', help="Device for PyTorch and model generation")
    parser.add_argument('--limit-model-concurrency', type=int, default=2, help="Maximum number of concurrent generator models working")
    parser.add_argument('--config-path', '-c', type=str, default="config.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    worker = ModelWorker(**vars(args))

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config['handlers']['default']['stream'] = log_config['handlers']['default']['stream'].replace('stderr', 'stdout')
    uvicorn.run(app, host=args.host, port=args.port, log_config=log_config, log_level='info')
