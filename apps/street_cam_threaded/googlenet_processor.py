#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# processes images via googlenet

from openvino.inference_engine import IENetwork, IEPlugin
import os
import numpy as np
import cv2
import queue
import threading


class googlenet_processor:
    EXAMPLES_BASE_DIR = '../../'
    ILSVRC_2012_dir = EXAMPLES_BASE_DIR + 'data/ilsvrc12/'
    LABELS_FILE_NAME = ILSVRC_2012_dir + 'synset_words.txt'

    # initialize the class instance
    # model is the file path of the googlenet IE model
    # plugin is an OpenVINO IEPlugin object
    # input_queue is a queue instance from which images will be pulled that are
    #     in turn processed (inferences are run on) via the NCS device
    #     each item on the queue should be an opencv image.  it will be resized
    #     as needed for the network
    # output_queue is a queue object on which the results of the inferences will be placed.
    #     For each inference a list of the following items will be placed on the output_queue:
    #         index of the most likely classification from the inference.
    #         label for the most likely classification from the inference.
    #         probability the most likely classification from the inference.
    def __init__(self, model: str, plugin: IEPlugin, input_queue: queue.Queue,
                 output_queue: queue.Queue, queue_wait_input: float, queue_wait_output: float):
        # labels to display along with boxes if googlenet classification is good
        # these will be read in from the synset_words.txt file for ilsvrc12
        self._gn_labels = [""]
        # loading the labels from file
        try:
            self._gn_labels = np.loadtxt(googlenet_processor.LABELS_FILE_NAME, str, delimiter='\t')
            for label_index in range(0, len(self._gn_labels)):
                temp = self._gn_labels[label_index].split(',')[0].split(' ', 1)[1]
                self._gn_labels[label_index] = temp
        except:
            print('\n\n')
            print('Error - could not read labels from: ' + googlenet_processor.LABELS_FILE_NAME)
            print('\n\n')
            raise

        weights = os.path.splitext(model)[0] + ".bin"
        assert os.path.isfile(model), "Cannot load input file %s" % model
        assert os.path.isfile(weights), "Cannot load input file %s" % weights
        net = IENetwork.from_ir(model=model, weights=weights)

        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        self._input_blob = next(iter(net.inputs))
        self._out_blob = next(iter(net.outputs))
        self._exec_net = plugin.load(network=net, num_requests=1)
        self._n, self._c, self._h, self._w = net.inputs[self._input_blob]
        del net

        self._queue_wait_input = queue_wait_input
        self._queue_wait_output = queue_wait_output

        self._input_queue = input_queue
        self._output_queue = output_queue
        self._worker_thread = threading.Thread(target=self._do_work, args=())

    # call one time when the instance will no longer be used.
    def cleanup(self):
        del self._exec_net

    # start asynchronous processing on a worker thread that will pull images off the input queue and
    # placing results on the output queue
    def start_processing(self):
        self._end_flag = False
        if (self._worker_thread == None):
            self._worker_thread = threading.Thread(target=self._do_work, args=())
        self._worker_thread.start()

    # stop asynchronous processing of the worker thread.
    # when returns the worker thread will have terminated.
    def stop_processing(self):
        self._end_flag = True
        self._worker_thread.join()
        self._worker_thread = None

    # the worker thread function. called when start_processing is called and
    # returns when stop_processing is called.
    def _do_work(self):
        print('in googlenet_processor worker thread')
        while (not self._end_flag):
            try:
                input_image = self._input_queue.get(True, self._queue_wait_input)
                index, label, probability = self.googlenet_inference(input_image, "NPS")
                self._output_queue.put((index, label, probability), True, self._queue_wait_output)
                self._input_queue.task_done()
            except queue.Empty:
                print('googlenet processor: No more images in queue.')
            except queue.Full:
                print('googlenet processor: queue full')

        print('exiting googlenet_processor worker thread')


    # Executes an inference using the googlenet graph and image passed
    # gn_graph is the googlenet graph object to use for the inference
    #   its assumed that this has been created with allocate graph and the
    #   googlenet graph file on an open NCS device.
    # input_image is the image on which a googlenet inference should be
    #   executed.  It will be resized to match googlenet image size requirements
    #   and also converted to float32.
    # returns a list of the following three items
    #   index of the most likely classification from the inference.
    #   label for the most likely classification from the inference.
    #   probability the most likely classification from the inference.
    def googlenet_inference(self, input_image:np.ndarray, user_obj):

        input_image = cv2.resize(input_image, (self._w, self._h), cv2.INTER_LINEAR)
        input_image = input_image.transpose((2, 0, 1)) # change data layout from HWC to CHW
        input_image = input_image.reshape((self._n, self._c, self._h, self._w))

        # Load tensor and get result.  This executes the inference on the NCS
        res = self._exec_net.infer({self._input_blob: input_image})
        output = res[self._out_blob][0]

        order = np.argsort(output)[-5:][::-1]

        '''
        print('\n------- prediction --------')
        for i in range(0, 5):
            print('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + self._gn_labels[
                order[i]] + '  label index is: ' + str(order[i]))
        '''

        # index, label, probability
        return order[0], self._gn_labels[order[0]], output[order[0]]
