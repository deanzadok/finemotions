from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from tracemalloc import start
sys.path.append('.')
sys.path.append('..')
import os
import time
import argparse
import numpy as np
from trainer.utils import ConfigManager
import onnxruntime as rt

parser = argparse.ArgumentParser()
parser.add_argument('--json', '-json', help='name of json file', default='config/mfm/test_mfm_aida_all_us2conf2multimidi.json', type=str)
args = parser.parse_args()
    
if __name__ == "__main__":

    # load config file
    cfg = ConfigManager(json_name=args.json)

    # check if output folder exists
    if not os.path.isdir(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # load model using onnx runtime
    sess = rt.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])

    # get input and output names
    output_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name

    # perform predictions
    durations = []
    for i in range(10000):
        input_img = np.zeros((cfg.training.batch_size, cfg.data.res, cfg.data.res, cfg.data.append), dtype=np.float32)
        start_time = time.time()
        predictions = sess.run([], {input_name: input_img})
        duration = time.time() - start_time
        durations.append(duration)
        
        if len(durations) > 30:
            print("FPS: {}".format(1/np.array(durations).mean()))
        print(predictions[1][-1,-1,:].tolist())    