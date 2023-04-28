import numpy as np
import sklearn
import skimage
from mindocr.postprocess.det_postprocess import PSEPostprocess

post_process = PSEPostprocess(binary_thresh=0, box_thresh=0.85, min_area=16,
                 box_type='quad', scale=1)

# pred = np.load("../debug/upsample_pred.npy")
pred = np.load("../debug/pred.npy")
post_process(pred)