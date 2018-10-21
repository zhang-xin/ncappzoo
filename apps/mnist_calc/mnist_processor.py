#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS, Heather McCabe
# Digit classifier using MNIST

import cv2
import numpy


class MnistProcessor:
    # The network assumes input images are these dimensions
    NETWORK_IMAGE_WIDTH = 28
    NETWORK_IMAGE_HEIGHT = 28

    def __init__(self, net, input_blob, out_blob, prob_thresh=0.0, classification_mask=None):
        """Initialize an instance of the class.

        :param net: the OpenVINO python ExecutableNetwork object
        :param net: the OpenVINO python network input blob
        :param net: the OpenVINO python network output blob
        :param prob_thresh: the probability threshold (between 0.0 and 1.0)... results below this threshold will be
        excluded
        :param classification_mask: a list of 0/1 values, one for each classification label in the
        _classification_mask list... if the value is 0 then the corresponding classification won't be reported.
        :return : None

        """
        # If no mask was passed then create one to accept all classifications
        self._exec_net = net
        self._input_blob = input_blob
        self._out_blob = out_blob
        self._classification_mask = classification_mask if classification_mask else [1] * 10
        self._probability_threshold = prob_thresh
        self._end_flag = True

    def _process_results(self, inference_result):
        """Interpret the output from a single inference of the neural network and filter out results with
        probabilities under the probability threshold.

        :param inference_result: the array of floats returned from the NCAPI as float32
        :return results: a list of 2-element sublists containing the detected digit as a string and the associated
        probability as a float

        """
        results = []

        print(inference_result)
        # Get a list of inference_result indexes sorted from highest to lowest probability
        sorted_indexes = (-inference_result).argsort()

        # Get a list of sub-lists containing the detected digit as an int and the probability
        for i in sorted_indexes:
            if inference_result[i] >= self._probability_threshold:
                results.append([i, inference_result[i]])
            else:
                # If this index had a value under the probability threshold, the rest of the indexes will too
                break

        return results

    def cleanup(self):
        pass

    @staticmethod
    def get_classification_labels():
        """Get a list of the classifications that are supported by this neural network.

        :return: the list of the classification strings
        """
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def do_sync_inference(self, input_image: numpy.ndarray):
        """Do a single inference synchronously.

        Don't mix this with calls to get_async_inference_result. Use one or the other. It is assumed
        that the input queue is empty when this is called which will be the case if this isn't mixed
        with calls to get_async_inference_result.

        :param input_image: the image on which to run the inference - it can be any size.
        :return: filtered results which is a list of lists. The inner lists contain the digit and its probability and
        are sorted from most probable to least probable.
        """
        res = self._exec_net.infer(inputs={self._input_blob: input_image})
        res = res[self._out_blob]
        results = self._process_results(res[0])

        return results

    @property
    def probability_threshold(self):
        return self._probability_threshold

    @probability_threshold.setter
    def probability_threshold(self, value):
        if 0.0 <= value <= 1.0:
            self._probability_threshold = value
        else:
            raise AttributeError('probability_threshold must be in range 0.0-1.0')
