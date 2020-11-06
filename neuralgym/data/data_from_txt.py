import random
import threading
import logging
import time

import cv2
import tensorflow as tf

from . import feeding_queue_runner as queue_runner
from .dataset import Dataset


READER_LOCK = threading.Lock()


class DataFromTxt(Dataset):
    """Data pipeline from list of filenames.

    Args:
        fnamelists (list): A list of filenames or tuple of filenames, e.g.
            ['image_001.png', ...] or
            [('pair_image_001_0.png', 'pair_image_001_1.png'), ...].
        shapes (tuple): Shapes of data, e.g. [256, 256, 3] or
            [[256, 256, 3], [1]].
        random (bool): Read from `fnamelists` randomly (default to False).
        random_crop (bool): If random crop to the shape from raw image or
            directly resize raw images to the shape.
        dtypes (tf.Type): Data types, default to tf.float32.
        enqueue_size (int): Enqueue size for pipeline.
        enqueue_size (int): Enqueue size for pipeline.
        nthreads (int): Parallel threads for reading from data.
        return_fnames (bool): If True, data_pipeline will also return fnames
            (last tensor).
        filetype (str): Currently only support image.

    Examples:
        >>> fnames = ['img001.png', 'img002.png', ..., 'img999.png']
        >>> data = ng.data.DataFromFNames(fnames, [256, 256, 3])
        >>> images = data.data_pipeline(128)
        >>> sess = tf.Session(config=tf.ConfigProto())
        >>> tf.train.start_queue_runners(sess)
        >>> for i in range(5): sess.run(images)

    To get file lists, you can either use file::

        with open('data/images.flist') as f:
            fnames = f.read().splitlines()

    or glob::

        import glob
        fnames = glob.glob('data/*.png')

    You can also create fnames tuple::

        with open('images.flist') as f:
            image_fnames = f.read().splitlines()
        with open('segmentation_annotation.flist') as f:
            annotation_fnames = f.read().splitlines()
        fnames = list(zip(image_fnames, annatation_fnames))

    """

    def __init__(self, fl_tuple, placeholder_info_list, decode_func,aug_funcs=None,random=False,
                 enqueue_size=32, queue_size=32, nthreads=16,**kwargs):
        self.placeholder_info_list = placeholder_info_list
        self.fl,self.process_fl_func = fl_tuple
        self.fl_list = self.process_fl_func(self.fl)
        self.file_length = len(self.fl_list)
        self.one_data_func = decode_func
        self.aug_funcs = aug_funcs

        self.batch_phs = [tf.placeholder(dtype,shape) for dtype, shape in placeholder_info_list]
        self.enqueue_size = enqueue_size
        self.queue_size = queue_size
        self.nthreads = nthreads
        self.index = 0
        self.kwargs = kwargs
        self.random = random
        super().__init__()
        self.create_queue()

    def data_pipeline(self, batch_size):
        """Batch data pipeline.

        Args:
            batch_size (int): Batch size.

        Returns:
            A tensor with shape [batch_size] and self.shapes
                e.g. if self.shapes = ([256, 256, 3], [1]), then return
                [[batch_size, 256, 256, 3], [batch_size, 1]].

        """
        data = self._queue.dequeue_many(batch_size)
        return data

    def create_queue(self, shared_name=None, name=None):
        from tensorflow.python.ops import data_flow_ops, logging_ops, math_ops
        from tensorflow.python.framework import dtypes
        capacity = self.queue_size
        self._queue = data_flow_ops.FIFOQueue(
            capacity=capacity,
            dtypes=[dtype for dtype, shape in self.placeholder_info_list],
            shapes=[shape[1:] for dtype, shape in self.placeholder_info_list]
            shared_name=shared_name,
            name=name)

        enq = self._queue.enqueue_many(self.batch_phs)
        # create a queue runner
        queue_runner.add_queue_runner(queue_runner.QueueRunner(
            self._queue, [enq]*self.nthreads,
            feed_dict_op=[lambda: self.next_batch()],
            feed_dict_key=self.batch_phs))
        # summary_name = 'fraction_of_%d_full' % capacity
        # logging_ops.scalar_summary("queue/%s/%s" % (
            # self._queue.name, summary_name), math_ops.cast(
                # self._queue.size(), dtypes.float32) * (1. / capacity))

    def next_batch(self):
        batch_data = []
        for _ in range(self.enqueue_size):
            while 1:
                if self.random:
                    if self.index % self.file_length == 0:
                        random.shuffle(self.fl_list)
                    self.index = (self.index + 1) % self.file_length
                    fl = self.fl_list[self.index]
                    # print(self.index);time.sleep(1)  ## for debug 
                else:
                    with READER_LOCK:
                        fl = self.fl_list[self.index]
                        self.index = (self.index + 1) % self.file_length
                try:
                    res = self.one_data_func(fl,**self.kwargs)
                    if self.aug_funcs is not None:
                        for aug_func in self.aug_funcs:
                            res = aug_func(res)
                except:
                    traceback.print_exc()
                    continue
                batch_data.append(res)
                break
        return zip(*batch_data)

    def _maybe_download_and_extract(self):
        pass
