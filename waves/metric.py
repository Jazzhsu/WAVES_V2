import os
import sys
from enum import Flag

import torch
from transformers import logging
import torch.multiprocessing as mp
from PIL import Image
import warnings
from tqdm.auto import tqdm
import dotenv

from waves.metrics import (
    compute_fid,
    compute_image_distance_repeated,
    load_perceptual_models,
    compute_perceptual_metric_repeated,
    load_aesthetics_and_artifacts_models,
    compute_aesthetics_and_artifacts_scores,
    load_open_clip_model_preprocess_and_tokenizer,
    compute_clip_score,
)
from waves.utils.image_loader import ImageLoader

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

#=======================================WAVES V2===================================================

class MetricOption(Flag):
    PSNR = 1 << 0
    SSIM = 1 << 1
    NMI = 1 << 2
    LPIPS = 1 << 3
    WATSON = 1 << 4
    AESTHETICS_ARTIFACTS = 1 << 5
    CLIP = 1 << 6

    SET_NONE = 0
    SET_ALL = 0xffffffff
    SET_CMP_METRICS = LPIPS | WATSON | PSNR | SSIM | NMI
    SET_SELF_METRICS = AESTHETICS_ARTIFACTS | CLIP

    @staticmethod
    def get_available_metrics():
        return [ metric for metric in MetricOption if not metric.name.startswith('SET_') ]

class DataOption(Flag):
    # Options for metrics that requires two data sets.
    CLEAN_VS_WATERMARKED = 1 << 0
    WATERMARKED_VS_ATTACKED = 1 << 1

    # Options for metrics that evaluate one data set.
    CLEAN = 1 << 2
    WATERMARKED = 1 << 3
    ATTACKED = 1 << 4

    NONE = 0

PATH_CLEAN = 'clean/'
PATH_WATERMARKED = 'wm/'
PATH_ATTACKED = 'attack/'

def _cmp_metric(im_loader_1: ImageLoader, im_loader_2: ImageLoader, metric: MetricOption, batch_size: int):
    """ Metrics that evalulate the difference of two image sets.

    TODO: Documentation.
    TODO: Fid implementation
    """
    if (len(im_loader_1) != len(im_loader_2)):
        raise ValueError(f'The size of the two image sets are different')

    im_loader_1.reset()
    im_loader_2.reset()

    # TODO: Better implementation.
    if metric in MetricOption.LPIPS | MetricOption.WATSON:
        mode = 'alex' if metric == MetricOption.LPIPS else 'dft'
        model = load_perceptual_models(metric.name.lower(), mode=mode)

    result = []
    while im_loader_1.has_next():
        im_set_1, im_set_2 = im_loader_1.load_batch(batch_size), im_loader_2.load_batch(batch_size)

        if metric in MetricOption.PSNR | MetricOption.SSIM | MetricOption.NMI:
            # TODO: Better implementation.
            result.extend(compute_image_distance_repeated(im_set_1, im_set_2, metric.name.lower(), num_workers=8, verbose=True))

        elif metric in MetricOption.LPIPS | MetricOption.WATSON:
            result.extend(compute_perceptual_metric_repeated(
                im_set_1,
                im_set_2,
                metric_name=metric.name.lower(),
                mode=mode,
                model=model,
                device=torch.device('cuda'),
            ))

    return {metric.name: result}

def _self_metric(im_loader: ImageLoader, prompts: list[str], metric: MetricOption, batch_size: int):
    """ Metrics that evaluate single image set.

    TODO: Documentation.
    """
    if len(im_loader) != len(prompts):
        raise ValueError(f'Size of the image sets and prompts do not match')

    im_loader.reset()

    # TODO: Better implementation
    if metric in MetricOption.AESTHETICS_ARTIFACTS:
        model = load_aesthetics_and_artifacts_models()
        result = {'aesthetics': [], 'artifacts': []}
    elif metric in MetricOption.CLIP:
        model = load_open_clip_model_preprocess_and_tokenizer()
        result = {metric.name: []}

    while im_loader.has_next():
        prompts_batch = prompts[im_loader.position():im_loader.position() + batch_size]
        images = im_loader.load_batch(batch_size)

        if metric in MetricOption.AESTHETICS_ARTIFACTS:
            aesthetics, artifacts = compute_aesthetics_and_artifacts_scores(images, model)
            result['aesthetics'].extend(aesthetics)
            result['artifacts'].extend(artifacts)

        elif metric in MetricOption.CLIP:
            result[metric.name].extend(compute_clip_score(images, prompts_batch, model))

    return result

def _select_data_source(clean_imgs: ImageLoader, wm_imgs: ImageLoader, attack_imgs: ImageLoader, option: DataOption):
    if option == DataOption.CLEAN:
        return clean_imgs
    if option == DataOption.WATERMARKED:
        return wm_imgs
    if option == DataOption.ATTACKED:
        return attack_imgs
    if option == DataOption.CLEAN_VS_WATERMARKED:
        return (clean_imgs, wm_imgs)
    if option == DataOption.WATERMARKED_VS_ATTACKED:
        return (wm_imgs, attack_imgs)

    raise ValueError(f'Unknown data option {option.name}.')

def generate_metrics(
    image_dir: str,
    prompts: list[str], # TODO: fix this.
    cmp_metric_option: MetricOption = MetricOption.SET_CMP_METRICS,
    cmp_metric_data_source: DataOption = DataOption.WATERMARKED_VS_ATTACKED | DataOption.CLEAN_VS_WATERMARKED,
    self_metric_option: MetricOption = MetricOption.SET_SELF_METRICS,
    self_metric_data_source: DataOption = DataOption.WATERMARKED | DataOption.ATTACKED,
    batch_size: int = 32,
):
    """ Generate selected metrics for selected data sets. This method serves as the main interface
    to calculate metrics.

    TODO: Documentations.
    """
    clean_imgs = ImageLoader(os.path.join(image_dir, PATH_CLEAN))
    wm_imgs = ImageLoader(os.path.join(image_dir, PATH_WATERMARKED))
    attacked_imgs = ImageLoader(os.path.join(image_dir, PATH_ATTACKED))

    available_metrics = MetricOption.get_available_metrics()

    # Self metrics
    result = {}
    for ds_option in [DataOption.CLEAN, DataOption.WATERMARKED, DataOption.ATTACKED]:
        if ds_option not in self_metric_data_source:
            continue

        result[ds_option.name] = {}
        loader = _select_data_source(clean_imgs, wm_imgs, attacked_imgs, ds_option)
        for metric in available_metrics:
            if metric not in self_metric_option:
                continue

            result[ds_option.name].update(_self_metric(loader, prompts, metric, batch_size))

    for ds_option in [DataOption.CLEAN_VS_WATERMARKED, DataOption.WATERMARKED_VS_ATTACKED]:
        if ds_option not in cmp_metric_data_source:
            continue

        result[ds_option.name] = {}
        loader1, loader2 = _select_data_source(clean_imgs, wm_imgs, attacked_imgs, ds_option)
        for metric in available_metrics:
            if metric not in cmp_metric_option:
                continue

            result[ds_option.name].update(_cmp_metric(loader1, loader2, metric, batch_size))

    return result

# TODO: Remove this when metric.py refactor is done.
TEST_PROMPT = [
    "Amidst the towering skyscrapers and congested streets of a metropolis, chaotic energy abounds. The scene is a collision of neon lights, honking vehicles, rushing pedestrians, and graffiti-covered walls, all contributing to an unrelenting urban frenzy.",
    "Amidst the towering skyscrapers and congested streets of a metropolis, chaotic energy abounds. The scene is a collision of neon lights, honking vehicles, rushing pedestrians, and graffiti-covered walls, all contributing to an unrelenting urban frenzy.",
    "Amidst the towering skyscrapers and congested streets of a metropolis, chaotic energy abounds. The scene is a collision of neon lights, honking vehicles, rushing pedestrians, and graffiti-covered walls, all contributing to an unrelenting urban frenzy.",
    "Amidst the towering skyscrapers and congested streets of a metropolis, chaotic energy abounds. The scene is a collision of neon lights, honking vehicles, rushing pedestrians, and graffiti-covered walls, all contributing to an unrelenting urban frenzy.",
    "In a serene and surreal tableau, an artist in all-white attire floats gracefully in a pool of milk. Her closed eyes and tranquil expression convey deep introspection as ripples radiate from their gentle movements, blurring the boundaries between reality and imagination.",
]

if __name__ == "__main__":
    """ Temporary code for generating attacked images. """
    # images = [ Image.open(f'test_image/wm_images/{i}.png') for i in range(5)]
    # images = distortions.apply_distortion(images, 'noise', 0.5)
    # for i, image in enumerate(images):
    #     image.save(f'test_image/attack_images/{i}.png')

    """ Expected image source file structure:
    test_image/
        |- clean/       ... Folder that stores clean images
        |- wm/          ... Folder that stores watermarked images
        |- attack/    ... Folder that stores attacked watermarked images
    """

    """ Example use case. """
    print(generate_metrics('test_image/', TEST_PROMPT, self_metric_data_source=DataOption.NONE, cmp_metric_data_source=DataOption.WATERMARKED_VS_ATTACKED))


