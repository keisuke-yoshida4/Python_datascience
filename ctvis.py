import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import nibabel as nib


def _get_df(base_path='public-covid-data', folder='rp_im'):
    data_dict = pd.DataFrame({'FilePath': glob('{}/{}/*'.format(base_path, folder)),
                'FileName': [p.split('/')[-1] for p in glob('{}/{}/*'.format(base_path, folder))]})
    return data_dict


def get_df_all(base_path='public-covid-data'):
    rp_im_df = _get_df(base_path, 'rp_im')
    rp_msk_df = _get_df(base_path, 'rp_msk')
    return rp_im_df.merge(rp_msk_df, on='FileName', suffixes=('Image', 'Mask'))


def load_nifti(path):
    nifti = nib.load(path)
    data = nifti.get_fdata()
    return np.rollaxis(data, 1, 0)


def label_color(mask_volume,
                ggo_color = [255, 0, 0],
                consolidation_color = [0, 255, 0],
                effusion_color = [0, 0, 255]):
    
    shp = mask_volume.shape
    # 箱作成
    mask_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32) 
    # 色付け
    mask_color[np.equal(mask_volume, 1)] = ggo_color
    mask_color[np.equal(mask_volume, 2)] = consolidation_color
    mask_color[np.equal(mask_volume, 3)] = effusion_color

    return mask_color


def hu_to_gray(volume):
    maxhu = np.max(volume)
    minhu = np.min(volume)
    volume_rerange = (volume - minhu) / max((maxhu - minhu), 1e-3)
    volume_rerange = volume_rerange * 255
    volume_rerange = np.stack([volume_rerange, volume_rerange, volume_rerange], axis=-1)
    
    return volume_rerange.astype(np.uint8)


def overlay(gray_volume, mask_volume, mask_color, alpha=0.3):
    mask_filter = np.greater(mask_volume, 0)
    mask_filter = np.stack([mask_filter, mask_filter, mask_filter], axis=-1)
    overlayed = np.where(mask_filter,
                         ((1-alpha)*gray_volume + alpha*mask_color).astype(np.uint8),
                         gray_volume)
    return overlayed


def vis_overlay(overlayed, original_volume, mask_volume, cols=5, display_num=25, figsize=(15, 15)):
    
    rows = (display_num - 1) // cols + 1
    total_num = overlayed.shape[-2]
    interval = total_num / display_num
    if interval < 1:
        interval = 1
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for i in range(display_num):
        row_i = i//cols
        col_i = i%cols
        idx = int((i * interval))
        if idx >= total_num:
            break
        stats = get_hu_stats(original_volume[:, :, idx], mask_volume[:, :, idx])
        title = 'slice #: {}'.format(idx)
        title += '\nggo_mean: {:.0f}±{:.0f}'.format(stats['ggo_mean'],stats['ggo_std'])
        title += '\nconsoli_mean: {:.0f}±{:.0f}'.format(stats['consolidation_mean'],stats['consolidation_std'])
        title += '\neffusion_mean: {:.0f}±{:.0f}'.format(stats['effusion_mean'],stats['effusion_std'])
        ax[row_i, col_i].imshow(overlayed[:, :, idx])
        ax[row_i, col_i].set_title(title)
        ax[row_i, col_i].axis('off')
    fig.tight_layout()
        
        
def get_hu_stats(volume,
                mask_volume,
                label_dict = {1: 'ggo', 2: 'consolidation', 3: 'effusion'}):
    
    result = {}

    for label in label_dict.keys():
        prefix = label_dict[label]
        roi_hu = volume[np.equal(mask_volume, label)]
        result[prefix + '_mean'] = np.mean(roi_hu)
        result[prefix + '_std'] = np.std(roi_hu)
        
    return result
    
    
