import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

##在TSNDataSet类的初始化函数 __init__ 中最重要的是self._parse_list()，也就是调用了 该类的_parse_list()方法。
##在该方法中，self.list_file就是训练或测试的列表文件（.txt文件），里面包含三列内容，用空格键分隔，
##第一列是video名，第二列是 video的帧数，第三列是video的标签。
##VideoRecord这个类只是提供了一些简单的封装，用来返回关于数据的一些信息（比如帧路径、该视频包含多少帧、帧标签）。
#因此最后self.video_list的内容就是一个长度为训练数据数量的列表，列表中的每个值都是VideoRecord对象，
##该对象包含一个列表和3个属性，列表长度为 3，分别是帧路径、该视频包含多少帧、帧标签，同样这三者也是三个属性的值。
class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='frame0{:05d}.jpg', transform=None,
                 ##image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()
##====================================================================
    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

##在TSNDataSet类的_sample_indices方法中，average_duration表示某个视频分成self.num_segments份的时候每一份包含多少帧图像
##因此只要该视频的总帧 数大于等于self.num_segments，就会执行if average_duration > 0这个条件，
##在该条件语句下offsets的计算分成两部分，np.multiply(list(range (self.num_segments)), average_duration)相当于确定了self.num_segments个片段的区间，
##randint(average_duration, size=self.num_segments)则是生成了 self.num_segments个范围在0到average_duration的数值，
##二者相加就相当于在这self.num_segments个片段中分别随机选择了一帧图像。
##因此在 __getitem__ 法中返回的segment_indices就是一个长度为self.num_segments的列表，表示帧的index。

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label-1

    def __len__(self):
        return len(self.video_list)
