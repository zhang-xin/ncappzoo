#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# Object detector using SSD Mobile Net

from openvino.inference_engine import IENetwork, IEPlugin, ExecutableNetwork
import numpy as numpy
import cv2
import time
import threading


class SsdMobileNetProcessor:

    def __init__(self, exec_net: ExecutableNetwork, n: int, c: int, w: int, h: int,
                 input_blob: str, out_blob: str, inital_box_prob_thresh: float,
                 classification_mask:list=None, name = None):
        """Initializes an instance of the class

        :param ncs_device: is an OpenVINO ExecutableNetwork object to use for inferences
        :param n, c, w, h: network input batch, channel, width, height
        :param input_blob, out_blob: network input output name
        :param inital_box_prob_thresh: the initial box probablity threshold. between 0.0 and 1.0
        :param classification_mask: a list of 0 or 1 values, one for each classification label in the
        :param name: A name to use for the processor.  Nice to have when debugging multiple instances
        on multiple threads
        :return : None
        """
        self._exec_net = exec_net
        self._n = n
        self._c = c
        self._w = w
        self._h = h
        self._input_blob = input_blob
        self._out_blob = out_blob
        self._box_probability_threshold = inital_box_prob_thresh
        self._classification_labels = SsdMobileNetProcessor.get_classification_labels()

        self._end_flag = True
        self._name = name
        if (self._name is None):
            self._name = "no name"

        # lock to let us count calls to asynchronus inferences and results
        self._async_count_lock = threading.Lock()
        self._async_inference_count = 0

        self._orig_image = {}

    def get_name(self):
        '''Get the name of this processor.

        :return:
        '''
        return self._name

    @staticmethod
    def get_classification_labels():
        """
        get a list of the classifications that are supported by this neural network
        :return: the list of the classification strings
        """
        ret_labels = [
            '', 'person', 'bicycle', 'car', 'motocycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',              # 00 - 10
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',                      # 11 - 20
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '',                                      # 21 - 30
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',  # 31 - 40
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',                # 41 - 50
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',                     # 51 - 60
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet',                                  # 61 - 70
            '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',                        # 71 - 80
            'sink', 'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'               # 81 - 90
        ]
        return ret_labels

    def start_aysnc_inference(self, input_image:numpy.ndarray):
        """Start an asynchronous inference.  When its complete it will go to the output FIFO queue which
           can be read using the get_async_inference_result() method
           If there is no room on the input queue this function will block indefinitely until there is room,
           when there is room, it will queue the inference and return immediately

        :param input_image: the image on which to run the inference.
             it can be any size but is assumed to be opencv standard format of BGRBGRBGR...
        :return: None
        """

        image = cv2.resize(input_image, (self._w, self._h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.reshape((self._n, self._c, self._h, self._w))

        self._inc_async_count()

        self._orig_image[0] = input_image
        # Load tensor. This executes the inference on the NCS
        self._exec_net.start_async(request_id=0, inputs={self._input_blob: image})

        return

    def _inc_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count += 1
        self._async_count_lock.release()

    def _dec_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count -= 1
        self._async_count_lock.release()

    def _get_async_count(self):
        self._async_count_lock.acquire()
        ret_val = self._async_inference_count
        self._async_count_lock.release()
        return ret_val


    def get_async_inference_result(self):
        """Reads the next available object from the output FIFO queue.  If there is nothing on the output FIFO,
        this fuction will block indefinitiley until there is.

        :return: tuple of the filtered results along with the original input image
        the filtered results is a list of lists. each of the inner lists represent one found object and contain
        the following 6 values:
           string that is network classification ie 'cat', or 'chair' etc
           float value for box X pixel location of upper left within source image
          float value for box Y pixel location of upper left within source image
          float value for box X pixel location of lower right within source image
          float value for box Y pixel location of lower right within source image
          float value that is the probability for the network classification 0.0 - 1.0 inclusive.
        """

        self._dec_async_count()

        ret = []
        if self._exec_net.requests[0].wait(-1) == 0:
            # Parse detection results of the current request
            res = self._exec_net.requests[0].outputs[self._out_blob]
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > self._box_probability_threshold:
                    height = self._orig_image[0].shape[0]
                    width = self._orig_image[0].shape[1]
                    xmin = int(obj[3] * width)
                    ymin = int(obj[4] * height)
                    xmax = int(obj[5] * width)
                    ymax = int(obj[6] * height)
                    class_id = int(obj[1])
                    ret.append([class_id, xmin, ymin, xmax, ymax, obj[2]])

        return ret, self._orig_image.get(0)


    def do_sync_inference(self, input_image:numpy.ndarray):
        """Do a single inference synchronously.
        Don't mix this with calls to get_async_inference_result, Use one or the other.  It is assumed
        that the input queue is empty when this is called which will be the case if this isn't mixed
        with calls to get_async_inference_result.

        :param input_image: the image on which to run the inference it can be any size.
        :return: filtered results which is a list of lists. Each of the inner lists represent one
        found object and contain the following 6 values:
            string that is network classification ie 'cat', or 'chair' etc
            float value for box X pixel location of upper left within source image
            float value for box Y pixel location of upper left within source image
            float value for box X pixel location of lower right within source image
            float value for box Y pixel location of lower right within source image
            float value that is the probability for the network classification 0.0 - 1.0 inclusive.
        """
        self.start_aysnc_inference(input_image)
        filtered_objects, original_image = self.get_async_inference_result()

        return filtered_objects


    def get_box_probability_threshold(self):
        """Determine the current box probabilty threshold for this instance.  It will be between 0.0 and 1.0.
        A higher number means less boxes will be returned.

        :return: the box probability threshold currently in place for this instance.
        """
        return self._box_probability_threshold


    def set_box_probability_threshold(self, value):
        """set the box probability threshold.

        :param value: the new box probability threshold value, it must be between 0.0 and 1.0.
        lower values will allow less certain boxes in the inferences
        which will result in more boxes per image.  Higher values will
        filter out less certain boxes and result in fewer boxes per
        inference.
        :return: None
        """
        self._box_probability_threshold = value

