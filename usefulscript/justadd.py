import argparse
import os

import numpy as np

from test_samples_onnx import test_onnx


def main(args):
    print("========== test session starts =========")
    model_name = os.path.basename(args.model_path).rsplit('.', 1)[0]
    print(f"test_samples_onnx.py::test_onnx[{model_name}]")

    test_onnx(model_name, args.model_path, args.exec_batch, args.max_batch, args.out_node, args.weight_share)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./justadd.rlym", help="onnx model file")
    # parser.add_argument('--model_dir', type=str, default=".", help="will find model in this directory if model path not exists")
    parser.add_argument('--exec_batch', type=int, choices=range(1, 65), default=1, help="set nne execute batch")
    parser.add_argument('--max_batch', type=int, choices=range(1, 65), default=32, help="set nne build engine max batch")
    parser.add_argument('--weight_share', choices=['0', '1', '2', '3', '01', '23', '0123'], default="0", help="set nne weight_share mode")
    parser.add_argument('--out_node', type=str, default="", help="set output node")
    # parser.add_argument('--out_node', type=str, default="resnetv15_conv0_fwd", help="set output node")
    args = parser.parse_args()
    main(args)

