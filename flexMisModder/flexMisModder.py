"""
flex imzML & mis file reader class
@author: Sebastian Krossa / MR Cancer / MH / ISB / NTNU Trondheim Norway
sebastian.krossa@ntnu.no
"""

import xml.etree.ElementTree as ET
import re
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from typing import NamedTuple

RegImage = NamedTuple("RegImage", [('path', str), ('tf', np.array), ('coreg_mis', bool), ('coreg_to', int)])
FlexRegion = NamedTuple("FlexRegion", [('name', str), ('points', np.array)])
CellType = NamedTuple('cell_type', [('id', int), ('single_cell', bool), ('area', float), ('bbox_p1', tuple), ('bbox_p2', tuple)])


class flexMisModder():
    def __init__(self, base_name, base_path, full_affine=True):
        self._mis_file = os.path.join(base_path, '{}.mis'.format(base_name))
        self.base_path = base_path
        if os.path.exists(self._mis_file):
            self._tree = ET.parse(self._mis_file)
            self._root = self._tree.getroot()
        else:
            raise ValueError("No mis file found!")

        self.imgs = self.extract_images(full_affine=full_affine)
        self.regions = self.extract_regions()
        self.mreg = self.get_mreg()
        self.main_img_scale = (np.sqrt(self.mreg[0,0]**2 + self.mreg[1,0]**2) +
                               np.sqrt(self.mreg[0,1]**2 + self.mreg[1,1]**2)) / 2

    def extract_images(self, root_xml=None, full_affine=True):
        if root_xml is None:
            root_xml = self._root
        _d = {}
        _img = None
        _base_img = None
        for c in root_xml:
            if c.tag == "CoRegistration":
                for cc in c:
                    _img = self.proc_child(cc, _d, _img)
                _d[_img]['CoReg'] = True
            elif c.tag in ["TeachPoint", "ImageFile"]:
                _img = self.proc_child(c, _d, _img)
                if _img is not None:
                    _base_img = _img
                _d[_img]['CoReg'] = False

        _rl = []
        if _base_img is not None:
            _r0, _r1 = self.extract_raster(root_xml)
            # y axis is inverted in machine
            _min_x, _ = np.array(_d[_base_img]['tps']).min(axis=0)
            _, _min_y = -1 * (np.array(_d[_base_img]['tps']).max(axis=0))
            for i, _tps in enumerate(_d[_base_img]['tps']):
                #_d[_base_img]['tps'][i] = [(_tps[0] - _min_x) / _r0, (-1 * _tps[1] - _min_y) / _r1]
                # do not devide by raster size to keep scale to µm and not msi pixel
                _d[_base_img]['tps'][i] = [(_tps[0] - _min_x), (-1 * _tps[1] - _min_y)]
        for _img, _dat in _d.items():
            if full_affine:
                _rl.append(RegImage(_img, cv2.getAffineTransform(np.array(_dat['ps']).astype(np.float32),
                                                                 np.array(_dat['tps']).astype(np.float32)),
                                    _dat['CoReg'], 0))
            else:
                _rl.append(RegImage(_img, cv2.estimateAffinePartial2D(np.array(_dat['ps']).astype(np.float32),
                                                                      np.array(_dat['tps']).astype(np.float32))[0],
                                    _dat['CoReg'], 0))
        return _rl

    @staticmethod
    def proc_child(c, _d, _img):
        if c.tag == "ImageFile":
            _img = c.text
            _d[_img] = {}
        if c.tag == "TeachPoint":
            raw = re.split(';|,', c.text)
            if 'ps' not in _d[_img]:
                _d[_img]['ps'] = []
                _d[_img]['tps'] = []
            _d[_img]['ps'].append([float(raw[0]), float(raw[1])])
            _d[_img]['tps'].append([float(raw[2]), float(raw[3])])
        return _img

    def get_mreg(self):
        for _img in self.imgs:
            if not _img.coreg_mis:
                return _img.tf

    def extract_regions(self, root_xml=None):
        if root_xml is None:
            root_xml = self._root
        _ret = []
        max_x_pt = [0, 0]
        max_y_pt = [0, 0]
        for item in root_xml:
            if item.tag in ['Area', 'ROI']:
                x = []
                y = []

                for child in item:
                    if child.tag == 'Point':
                        raw_vals = child.text.split(',')
                        # print(raw_vals)
                        x.append(int(raw_vals[0]))
                        y.append(int(raw_vals[1]))
                        if int(raw_vals[0]) > max_x_pt[0]:
                            max_x_pt = [int(raw_vals[0]), int(raw_vals[1])]
                        if int(raw_vals[1]) > max_y_pt[1]:
                            max_y_pt = [int(raw_vals[0]), int(raw_vals[1])]
                _ret.append(FlexRegion(item.attrib['Name'], np.array((x, y)).T))
        return _ret

    @staticmethod
    def extract_raster(root_xml):
        for area in root_xml.iter('Area'):
            for child in area:
                if child.tag == 'Raster':
                    _r1, _r2 = child.text.split(',')
                    return int(_r1), int(_r2)

    def get_transformed_regions(self):
        _rd = {}
        for _reg in self.regions:
            _rd[_reg.name] = FlexRegion(_reg.name, self.transform(_reg.points, self.mreg))
        return _rd

    def get_transformed_images(self):
        _rd = {}
        for _img in self.imgs:
            _img_p = None
            _target_img = None
            if _img.coreg_mis:
                _tm = np.dot(np.vstack([self.mreg, np.array([0, 0, 1])]),
                             np.vstack([cv2.invertAffineTransform(_img.tf), np.array([0, 0, 1])]))[:2, :]
            elif _img.coreg_to > 0:
                _tm = np.dot(np.dot(np.vstack([self.mreg, np.array([0, 0, 1])]),
                                    np.vstack([cv2.invertAffineTransform(self.imgs[_img.coreg_to].tf),
                                               np.array([0, 0, 1])])), _img.tf)[:2, :]
                _img_p = _img.path
                _target_img = cv2.imread(os.path.join(self.base_path, self.imgs[_img.coreg_to].path),
                                         cv2.IMREAD_UNCHANGED)
                _ttm = np.dot(np.vstack([self.mreg, np.array([0, 0, 1])]),
                              np.vstack([cv2.invertAffineTransform(self.imgs[_img.coreg_to].tf),
                                         np.array([0, 0, 1])]))[:2, :]
            else:
                _tm = self.mreg
            if _img_p is None:
                _img_p = os.path.join(self.base_path, _img.path)
            _imgo = plt.imread(_img_p)
            if _target_img is None:
                _target_img = _imgo
                _ttm = _tm
            _w, _h = tuple(np.ceil(self.transform([(_target_img.shape[1], _target_img.shape[0])], _ttm)).astype(int)[0])
            _rd['tf_{}'.format(_img.path)] = cv2.warpAffine(_imgo, _tm, (_w, _h))
        return _rd

    @staticmethod
    def is_inside_cnt(cnt, x, y):
        if cv2.pointPolygonTest(cnt.round().astype(np.int32), (x, y), True) >= 0:
            return True
        else:
            return False

    @staticmethod
    def is_inside_cnts(cnts, x, y):
        if isinstance(cnts, list):
            for _c in cnts:
                if flexMisModder.is_inside_cnt(_c, x, y):
                    return True
            return False
        elif isinstance(cnts, np.ndarray):
            return flexMisModder.is_inside_cnt(cnts, x, y)
        else:
            return False

    @staticmethod
    def _use_point(x, y, unique_x, unique_y, cnt):
        if cnt is None:
            return x in unique_x and y in unique_y
        else:
            return flexMisModder.is_inside_cnts(cnt, x, y)

    def get_regions_max_xy(self):
        m_x = np.array([0, 0])
        m_y = np.array([0, 0])
        for _reg in self.regions:
            for e in _reg.points:
                if e[0] > m_x[0]:
                    m_x = e
                if e[1] > m_y[1]:
                    m_y = e
        return m_x, m_y

    @staticmethod
    def transform(points, mtx):
        tmp = []
        for p in points:
            tmp.append(np.dot(mtx[0:2, 0:2], p) + mtx[0:2, 2])
        return np.array(tmp)

    @staticmethod
    def _identity_norm(x, y):
        return x

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    @staticmethod
    def simplify_contour(cnt, simplify_factor=0.001):
        cnt = np.round(cnt).astype(int)
        return cv2.approxPolyDP(cnt, simplify_factor * cv2.arcLength(cnt, True), True)[:, 0, :]

    def inject_contour_into_mis(self, cnt, name, color='#000000', mtf=None, simplify=True, simplify_factor=0.001,
                                save=True):
        roi = ET.SubElement(self._root, 'ROI')
        roi.set('Type', '3')
        roi.set('Name', name)
        roi.set('Enabled', '0')
        roi.set('ShowSpectra', '0')   
        roi.set('SpectrumColor', color)
        if mtf is not None:
            if mtf == 'auto':
                cnt = self.transform(cnt,
                                     np.linalg.inv(np.vstack([self.mreg, np.array([0, 0, 1])])))
            else:
                cnt = self.transform(cnt, mtf)
        cnt = np.round(cnt).astype(int)
        if simplify:
            cnt = flexMisModder.simplify_contour(cnt=cnt, simplify_factor=simplify_factor)
        for x, y in cnt:
            _p = ET.SubElement(roi, 'Point')
            _p.text = '{},{}'.format(x, y)
        if save:
            self.save_mis_file()
        return roi, cnt
    
    def inject_bb_into_mis(self, bb, name, color='#000000', save=True):
        roi = ET.SubElement(self._root, 'ROI')
        roi.set('Type', '3')
        roi.set('Name', name)
        roi.set('Enabled', '0')
        roi.set('ShowSpectra', '0')
        roi.set('SpectrumColor', color)
        for x, y in zip([bb[0][0], bb[1][0], bb[1][0], bb[0][0]], [bb[0][1], bb[0][1], bb[1][1], bb[1][1]]):
            _p = ET.SubElement(roi, 'Point')
            _p.text = '{},{}'.format(x, y)
        if save:
            self.save_mis_file()
        return roi, bb
    
    def save_mis_file(self, filename_mod='_mod'):
        self._tree.write('{}'.format(filename_mod).join(os.path.splitext(self._mis_file)))
        return None

    @staticmethod
    def get_mask(img, use_hist_equal=False, use_otsu=True, use_adaptive_thr=False, bin_thr_cut=False):
        ret = None
        _gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if use_hist_equal:
            _gray = cv2.equalizeHist(_gray)
        if use_otsu:
            _blur = cv2.GaussianBlur(_gray, (5, 5), 0)
            ret, _th1 = cv2.threshold(_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _blur = cv2.GaussianBlur(_gray, (5, 5), 0)
            if use_adaptive_thr:
                _th1 = cv2.adaptiveThreshold(_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 2)
            else:
                ret, _th1 = cv2.threshold(_blur, bin_thr_cut, 255, cv2.THRESH_BINARY)
        return ret, _th1

    @staticmethod
    def detect_cells(mask, mask_scale_um_per_px, filter_shape_low=0.25, filter_shape_high=4, single_shape_low=0.75,
                     single_shape_high=1.25, max_cell_a_um=10000, min_cell_a_um=100, single_cell_max_box_um=2500,
                     um_bb_margin=10, mask_x_offset=0, mask_y_offset=0):
        px_off = int(round(um_bb_margin / mask_scale_um_per_px))
        _contours, _hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cells = []
        for i, _c in enumerate(_contours):
            x, y, w, h = cv2.boundingRect(_c)
            x1 = x + mask_x_offset - px_off
            y1 = y + mask_y_offset - px_off
            x2 = x1 + w + 2 * px_off
            y2 = y1 + h + 2 * px_off
            a_um = w * h * mask_scale_um_per_px ** 2
            # "coarse" filter of contour by aspect ratio and size of bounding box
            if (filter_shape_low < (w / h) < filter_shape_high) \
                    and h > 0 and w > 0 and min_cell_a_um <= a_um <= max_cell_a_um:
                if a_um <= single_cell_max_box_um and (single_shape_low < (w / h) < single_shape_high):
                    _single_cell = True
                else:
                    _single_cell = False
                cells.append(CellType(id=i, single_cell=_single_cell, area=a_um, bbox_p1=(x1, y1), bbox_p2=(x2, y2)))
        return cells

    @staticmethod
    def draw_cells_on_img(cells, img):
        for cell in cells:
            if cell.single_cell:
                _c = (0, 255, 0)
            else:
                _c = (0, 0, 255)
            cv2.rectangle(img, cell.bbox_p1, cell.bbox_p2, _c, 1)
        return img

    def inject_cells_bb_into_mis(self, cells, save=True, single_cell_color='#00FF00', multi_cell_color='#0000FF'):
        for cell in cells:
            if cell.single_cell:
                color = single_cell_color
                name = 'potential_single_cell_id_{}'.format(cell.id)
            else:
                color = multi_cell_color
                name = 'potential_multiple_cells_id_{}'.format(cell.id)

            _roi, bb = self.inject_bb_into_mis(bb=[cell.bbox_p1, cell.bbox_p2], name=name, color=color, save=False)
        if save:
            self.save_mis_file()