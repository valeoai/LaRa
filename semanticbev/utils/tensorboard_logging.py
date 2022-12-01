import matplotlib.pyplot as plt
import itertools
import torchvision
import torch
import numpy as np

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


def gridplot(imgs, titles=[], cmaps=[], cols=2, figsize=(12, 12)):
    """
    Plot a list of images in a grid format

    :param imgs: list of images to plot
    :param titles: list of titles to print above each image of `imgs`
    :param cols: number of column of the grid, the number of rows is determined accordingly
    :param figsize: matplotlib `figsize` figure param
    """

    rows = len(imgs) // cols + len(imgs) % cols

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()

    for img, title, cmap, ax in itertools.zip_longest(imgs, titles, cmaps, axs):

        if img is None:
            ax.set_visible(False)
            continue

        if img.ndim == 2 and cmap is None:
            cmap = 'gray'

        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    return fig


def prep_image(image):
    """
    Prepare image for tensorboard logging
    Parameters
    ----------
    image : torch.Tensor [3,H,W]
        Image to be logged
    Returns
    -------
    image : numpy array
        image as numpy array
    """
    image = image.detach().permute(1, 2, 0).cpu().numpy()

    return image


def prep_visibility_map(visibility):

    v = visibility.clone().int()[0]
    visibility_map = torch.zeros(3, 200, 200, dtype=int)

    # visibility=1  visibility of whole object is between and 0 and 40% (red)
    visibility_map[0, :, :][v == 1] = 255

    # visibility=2  visibility of whole object is between and 40 and 60% (green)
    visibility_map[1, :, :][v == 2] = 255

    # visibility=3  visibility of whole object is between and 60 and 80% (cyan)
    visibility_map[1, :, :][v == 3] = 255
    visibility_map[2, :, :][v == 3] = 255

    # visibility=4  visibility of whole object is between and 80 and 100% (yellow)
    visibility_map[0, :, :][v == 4] = 255
    visibility_map[1, :, :][v == 4] = 255

    visibility_map[0, :, :][v == 255] = 0
    visibility_map[1, :, :][v == 255] = 0
    visibility_map[2, :, :][v == 255] = 0

    return visibility_map


def prepare_images_to_log(learning_phase, batch, preds, batch_idx, log_images_interval):
    if log_images_interval == 0 or batch_idx % log_images_interval != 0:
        return {}

    i = 0  # for each batch, we always log the first image of the batch only

    prefix = f"{learning_phase}/batch{batch_idx}"

    visibility_map = prep_visibility_map(batch['visibility'][i])

    img_list = [
        prep_image(preds[i].sigmoid()),
        prep_image(visibility_map)
    ]
    titles = ['pred', 'gt']
    cmaps = ['gray', 'gray']

    for cam_idx, cam in enumerate(cams):
        if batch['imgs'].dim() == 6:
            cam_img = batch['imgs'][i, -1, cam_idx, :, :, :]
        else:
            cam_img = batch['imgs'][i, cam_idx, :, :, :]

        img_list.append(np.array(denormalize_img(cam_img)))
        titles.append(cam)
        cmaps.append(None)

    plt_figure = gridplot(img_list, titles=titles, cmaps=cmaps, cols=4, figsize=(6 * 4, 6 * ((len(titles) + 1) // 4)))

    return {prefix: plt_figure}

