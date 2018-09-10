#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from os import listdir
from os.path import isfile, join
import random
from argparse import ArgumentParser
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin


random.seed()


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-i", "--input", help="Path to a folder with images or path to an image files", required=True,
                        type=str)
    parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    parser.add_argument("--hello", help="Open device but do not do inference", action="store_true")
    parser.add_argument("--gui", help="Show Window for each iteration", action="store_true")
    parser.add_argument("-nt", "--number_top", help="Number of top results", default=5, type=int)
    parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=100, type=int)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device="MYRIAD")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    n, c, h, w = net.inputs[input_blob]
    assert n == 1, "Sample supports batch size is 1 only"

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    del net

    if args.hello:
        return

    # Read and pre-process input images
    selected_files = [join(args.input, f) for f in listdir(args.input) if isfile(join(args.input, f))]
    selected_files = selected_files[:100]
    imgarr = []
    for img_f in selected_files:
        images = np.ndarray(shape=(n, c, h, w))
        image_clone = None
        for i in range(n):
            image = cv2.imread(img_f)
            image_clone = image
            if image.shape[:-1] != (h, w):
                image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[i] = image
        imgarr.append([img_f, images, image_clone])

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(args.number_iter))
    infer_time = []
    for i in range(args.number_iter):
        img_idx = random.randint(0, len(imgarr) - 1)
        frame = imgarr[img_idx][2]
        t0 = time()
        infer_request_handle = exec_net.start_async(request_id=0, inputs={input_blob: imgarr[img_idx][1]})
        infer_request_handle.wait()
        infer_time.append((time() - t0) * 1000)

        # Processing output blob
        log.info("Processing output blob")
        res = infer_request_handle.outputs[out_blob]
        log.info("On image {}".format(imgarr[img_idx][0]))
        log.info("Top {} results: ".format(args.number_top))
        if args.labels:
            with open(args.labels, 'r') as f:
                labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        else:
            labels_map = None
        for _i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)[-args.number_top:][::-1]
            for id in top_ind:
                det_label = labels_map[id] if labels_map else "#{}".format(id)
                print("{:.7f} {}".format(probs[id], det_label))
                if args.gui:
                    cv2.putText(frame, "{:.7f} {}: ".format(probs[id], det_label), (15, 30+(id+1)*10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            print("\n")

        if args.gui:
            cv2.putText(frame, "Inference time: {} ms".format(infer_time[len(infer_time) - 1]), (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, "Top {} results: ".format(args.number_top), (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.imshow("Classification Results", frame)
            key = cv2.waitKey(0)
            if key == 27:
                break
            else:
                continue

    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))

    cv2.destroyAllWindows()
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
