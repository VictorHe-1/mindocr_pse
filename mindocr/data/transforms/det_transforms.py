"""
transforms for text detection tasks.
"""
import random
import warnings
from typing import List

import json
import cv2
import pyclipper
from shapely.geometry import Polygon
import numpy as np
from copy import deepcopy
import sys

__all__ = ['DetLabelEncode', 'PSEGtDecode', 'BorderMap', 'ShrinkBinaryMap', 'expand_poly', 'DetResizeForTest']


class DetLabelEncode:
    def __init__(self, **kwargs):
        pass

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    def __call__(self, data):
        """
        required keys:
            label (str): string containgin points and transcription in json format
        added keys:
            polys (np.ndarray): polygon boxes in an image, each polygon is represented by points
                            in shape [num_polygons, num_points, 2]
            texts (List(str)): text string
            ignore_tags (np.ndarray[bool]): indicators for ignorable texts (e.g., '###')
        """
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data


# FIXME:
#  RuntimeWarning: invalid value encountered in sqrt result = np.sqrt(a_sq * b_sq * sin_sq / c_sq)
#  RuntimeWarning: invalid value encountered in true_divide cos = (a_sq + b_sq - c_sq) / (2 * np.sqrt(a_sq * b_sq))
warnings.filterwarnings("ignore")
class BorderMap:
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self._shrink_ratio = shrink_ratio
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max
        self._dist_coef = 1 - self._shrink_ratio ** 2

    def __call__(self, data):
        border = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.zeros(data['image'].shape[:2], dtype=np.float32)

        for i in range(len(data['polys'])):
            if not data['ignore_tags'][i]:
                self._draw_border(data['polys'][i], border, mask=mask)
        border = border * (self._thresh_max - self._thresh_min) + self._thresh_min

        data['thresh_map'] = border
        data['thresh_mask'] = mask
        return data

    def _draw_border(self, np_poly, border, mask):
        # draw mask
        poly = Polygon(np_poly)
        distance = self._dist_coef * poly.area / poly.length
        padded_polygon = np.array(expand_poly(np_poly, distance)[0], dtype=np.int32)
        cv2.fillPoly(mask, [padded_polygon], 1.0)

        # draw border
        min_vals, max_vals = np.min(padded_polygon, axis=0), np.max(padded_polygon, axis=0)
        width, height = max_vals - min_vals + 1
        np_poly -= min_vals

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = [self._distance(xs, ys, p1, p2) for p1, p2 in zip(np_poly, np.roll(np_poly, 1, axis=0))]
        distance_map = np.clip(np.array(distance_map, dtype=np.float32) / distance, 0, 1).min(axis=0)   # NOQA

        min_valid = np.clip(min_vals, 0, np.array(border.shape[::-1]) - 1)  # shape reverse order: w, h
        max_valid = np.clip(max_vals, 0, np.array(border.shape[::-1]) - 1)

        border[min_valid[1]: max_valid[1] + 1, min_valid[0]: max_valid[0] + 1] = np.fmax(
            1 - distance_map[min_valid[1] - min_vals[1]: max_valid[1] - max_vals[1] + height,
                             min_valid[0] - min_vals[0]: max_valid[0] - max_vals[0] + width],
            border[min_valid[1]: max_valid[1] + 1, min_valid[0]: max_valid[0] + 1]
        )

    @staticmethod
    def _distance(xs, ys, point_1, point_2):
        """
        compute the distance from each point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        a_sq = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        b_sq = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        c_sq = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cos = (a_sq + b_sq - c_sq) / (2 * np.sqrt(a_sq * b_sq))
        sin_sq = np.nan_to_num(1 - np.square(cos))
        result = np.sqrt(a_sq * b_sq * sin_sq / c_sq)

        result[cos >= 0] = np.sqrt(np.fmin(a_sq, b_sq))[cos >= 0]
        return result



class ShrinkBinaryMap:
    """
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """
    def __init__(self, min_text_size=8, shrink_ratio=0.4, train=True):
        self._min_text_size = min_text_size
        self._shrink_ratio = shrink_ratio
        self._train = train
        self._dist_coef = 1 - self._shrink_ratio ** 2

    def __call__(self, data):
        gt = np.zeros(data['image'].shape[:2], dtype=np.float32)
        mask = np.ones(data['image'].shape[:2], dtype=np.float32)

        if len(data['polys']):
            if self._train:
                self._validate_polys(data)

            for i in range(len(data['polys'])):
                min_side = min(np.max(data['polys'][i], axis=0) - np.min(data['polys'][i], axis=0))

                if data['ignore_tags'][i] or min_side < self._min_text_size:
                    cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                    data['ignore_tags'][i] = True
                else:
                    poly = Polygon(data['polys'][i])
                    shrunk = expand_poly(data['polys'][i], distance=-self._dist_coef * poly.area / poly.length)

                    if shrunk:
                        cv2.fillPoly(gt, [np.array(shrunk[0], dtype=np.int32)], 1)
                    else:
                        cv2.fillPoly(mask, [data['polys'][i].astype(np.int32)], 0)
                        data['ignore_tags'][i] = True

        data['binary_map'] = np.expand_dims(gt, axis=0)
        data['mask'] = mask
        return data

    @staticmethod
    def _validate_polys(data):
        data['polys'] = np.clip(data['polys'], 0, np.array(data['image'].shape[1::-1]) - 1)  # shape reverse order: w, h

        for i in range(len(data['polys'])):
            poly = Polygon(data['polys'][i])
            if poly.area < 1:
                data['ignore_tags'][i] = True
            if not poly.exterior.is_ccw:
                data['polys'][i] = data['polys'][i][::-1]


def expand_poly(poly, distance: float, joint_type=pyclipper.JT_ROUND) -> List[list]:
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(poly, joint_type, pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(distance)

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri
def shrink(text_polys, rate, max_shr=20):
    rate = rate * rate
    shrinked_text_polys = []
    for bbox in text_polys:
        area = Polygon(bbox).area
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

        shrinked_bbox = pco.Execute(-offset) # (N, 2) shape, N maybe larger than or smaller than 4.
        if not shrinked_bbox:
            shrinked_text_polys.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_text_polys.append(bbox)
            continue

        shrinked_text_polys.append(shrinked_bbox)

    return shrinked_text_polys

class PSEGtDecode(object):
    def __init__(self, kernel_num=7, min_shrink_ratio=0.4, min_shortest_edge=640):
        self.kernel_num = kernel_num
        self.min_shrink_ratio = min_shrink_ratio
        self.min_shortest_edge = min_shortest_edge

    def __call__(self, data):

        image = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']

        h, w, _ = image.shape
        short_edge = min(h, w) 
        if short_edge < self.min_shortest_edge:
            # keep short_size >= self.min_short_edge
            scale = self.min_shortest_edge / short_edge
            image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
            text_polys *= scale

        # get gt_text and training_mask
        img_h, img_w = image.shape[0: 2]
        gt_text = np.zeros((img_h, img_w), dtype=np.float32)
        training_mask = np.ones((img_h, img_w), dtype=np.float32)
        if text_polys.shape[0] > 0:
            text_polys = text_polys.astype('int32')
            for i in range(text_polys.shape[0]):
                cv2.drawContours(gt_text, [text_polys[i]], 0, i + 1, -1)
                if ignore_tags[i]:
                    cv2.drawContours(training_mask, [text_polys[i]], 0, 0, -1)

        # get gt_kernels
        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_shrink_ratio) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros((img_h, img_w), dtype=np.float32)
            kernel_text_polys = shrink(text_polys, rate)
            for j in range(len(kernel_text_polys)):
                cv2.drawContours(gt_kernel, [kernel_text_polys[j]], 0, 1, -1)
            gt_kernels.append(gt_kernel)

        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        data['image'] = image
        data['polys'] = text_polys
        data['gt_kernels'] = gt_kernels
        data['gt_text'] = gt_text
        data['mask'] = training_mask
        return data

class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_short' in kwargs:
            self.limit_side_len = 736
            self.limit_type = 'min'
        else:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)

    def __call__(self, data):
        # img = deepcopy(data)
        img = deepcopy(data['image'])
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['image_shape'] = (src_h, src_w, ratio_h, ratio_w)
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # return img, np.array([h, w])
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]