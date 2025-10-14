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

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover():
    def __init__(self, model_name='isnet-general-use'):
        self.session = new_session(model_name=model_name)

    def __call__(
        self, image: Image.Image, alpha_matting=False, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_size=10,
        **kwargs,
    ):
        """
        Remove background from image.

        Parameters
        ----------
        image : Image
            Input image.
        alpha_matting : bool, optional, default=False
            Alpha matting.
        alpha_matting_foreground_threshold : int, optional, default=240
            Alpha matting foreground threshold.
        alpha_matting_background_threshold : int, optional, default=10
            Alpha matting background threshold.
        alpha_matting_erode_size : int, optional, default=10
            Alpha matting erode size.

        Returns
        -------
        output : btyes | Image | ndarray
            Cutout image with background removed.
        """

        output = remove(
            image, alpha_matting=alpha_matting, alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold, alpha_matting_erode_size=alpha_matting_erode_size,
            session=self.session, bgcolor=(255, 255, 255, 0),
        )
        return output
