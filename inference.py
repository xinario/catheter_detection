from options.test_options import TestOptions
from models.scale_rcnn import SRCNN
from PIL import Image
import io
from data.preprocess import preprocess_single_image
from types import SimpleNamespace
import math
import numpy as np
import torch
import torchvision.transforms as transforms


def get_inference_options():
    opt = SimpleNamespace(
        dataroot="./datasets/pediatric_internet/",
        batchSize=1,
        loadSize=480,
        fineSize=512,
        input_nc=3,
        output_nc=3,
        ngf=64,
        which_model_netG="srcnn",
        gpu_ids="",
        name="catheter_detect",
        dataset_mode="alignedsrcnn",
        model="srcnn",
        which_direction="AtoB",
        nThreads=0,
        checkpoints_dir="./checkpoints",
        norm="instance",
        serial_batches=False,
        display_winsize=256,
        display_id=1,
        display_port=8097,
        no_dropout=False,
        max_dataset_size=math.inf,
        resize_or_crop="none",
        no_flip=True,
        init_type="normal",
        ntest=math.inf,
        results_dir="./results/",
        aspect_ratio=1.0,
        phase="test",
        which_epoch="latest",
        how_many=100,
        sourceoftest="internal",
        isTrain=False,
    )
    return opt


def segment_xray(image_bytes):
    """
    Function to segment X-ray images using a pre-trained SRCNN model.
    Args:
        image_bytes (bytes): The input image in bytes format.
    Returns:
        PIL.Image: The segmented image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    opt = get_inference_options()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    model = SRCNN()
    model.initialize(opt)

    tensors = preprocess_single_image(image, fine_size=480)

    model.set_input(tensors)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    # for label, image_numpy in visuals.items():
    #     print(image_numpy.shape)
    #     util.save_image(image_numpy, f"./test_output_{label}.png")
    # mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(visuals["fake_B2"])  # mode = "L" (grayscale)
    return mask_img


def overlay_mask_on_image(original: Image.Image, mask: np.ndarray) -> Image.Image:
    """Overlay a binary mask onto the original image with red transparency."""
    original = original.convert("RGBA").resize(mask.shape[::-1])
    overlay = Image.new("RGBA", original.size, (255, 0, 0, 128))  # red mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(original.size)
    return Image.composite(overlay, original, mask_img.convert("L"))


# segment_xray(open("/Users/xin/Documents/past_research/test8.jpg", "rb").read())


def classify_xray(image_bytes):
    """
    Function to classify X-ray images.
    Args:
        image_bytes (bytes): The input image in bytes format.
    Returns:
        str: Classification result.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Grayscale
    print("| Loading checkpoint model for test phase...")
    checkpoint = torch.load(
        "./checkpoints/catheter_classification/resnet-50.t7",
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    model = checkpoint["model"]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()

    model.eval()

    data_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input = data_transform(image)
    input = input.unsqueeze(0)  # Add batch dimension
    outputs = model(input)
    preds = torch.sigmoid(outputs)

    return preds  # Example placeholder result
