
try:
    import wandb
except ImportError:
    raise ImportError('You want to use `wandb` logger which is not installed yet,'
                      ' install it with `pip install wandb`.')

cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def prep_image(image, caption=None):
    image = image.detach().permute(1, 2, 0).cpu().numpy()

    return wandb.Image(image, caption=caption)


def prepare_images_to_log(learning_phase, batch, binimgs, preds, batch_idx, log_images_interval):
    if log_images_interval == 0 or batch_idx % log_images_interval != 0:
        return {}

    i = 0  # for each batch, we always log the first image of the batch only

    prefix = f"{learning_phase}/batch{batch_idx}"

    img_list = [
        prep_image(preds[i].sigmoid(), caption='pred'),
        prep_image(binimgs[i], caption='gt')
    ]

    for cam_idx, cam in enumerate(cams):
        img_list.append(prep_image(batch['imgs'][i, cam_idx, :, :, :], caption=cam))

    return {prefix: img_list}

