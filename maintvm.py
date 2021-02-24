import argparse
import random
import pdb, os, sys, time

import torch
import torchvision
import numpy as np

import tvm
import tvm.target
import tvm.contrib.ndk
import tvm.contrib.utils
import tvm.contrib.graph_runtime
from tvm import relay, rpc, te

from model import EfficientDet, EfficientDetD3



def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('-mode', choices=['trainval', 'eval'], default='eval', type=str)
    parser.add_argument('-model', default='efficientdet-d3', type=str)
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    parser.add_argument('--device', type=int, default=0)
    parser.set_defaults(cuda=False)
    return parser.parse_args(args)


def main(args):
    # TVM_NDK_CC="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android27-clang++"
    assert "TVM_NDK_CC" in os.environ
    
    device = torch.device('cuda:{}'.format(args.device)) \
        if args.cuda else torch.device('cpu')

    model = EfficientDet.from_pretrained(args.model).to(device) \
        if args.pretrained else EfficientDet.from_name(args.model).to(device)
    model_pnames = '\n'.join(sorted([n+': ('+', '.join([str(p.size(i))
                                                       for i in range(p.ndim)])+')'
                                    for n,p in model.named_parameters()]))
    model = EfficientDetD3.from_pretrained().to(device) \
        if args.pretrained else EfficientDetD3.from_name().to(device)
    model2_pnames = '\n'.join(sorted([n+': ('+', '.join([str(p.size(i))
                                                         for i in range(p.ndim)])+')'
                                      for n,p in model.named_parameters()]))
    print(model_pnames == model2_pnames)
    #pdb.set_trace()

    #a,b = model(torch.randn(1,3,896,896))
    #for i in range(5):
    #    print(a[i].size(), b[i].size())
    
    
    print("Tracing PyTorch model into TorchScript...")
    model = model.eval()
    input_shape = [1, 3, 896, 896]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    
    print("Analyzing PyTorch model with Relay...")
    input_name = "input"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    
    print("Compiling library for Kryo CPU...")
    target = tvm.target.arm_cpu("kryo")
    target_host = "llvm -mtriple=aarch64-unknown-linux-android27 -mcpu=kryo -model=kryo"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    
    print("Exporting deploy_lib.so binary to disk...")
    ret = lib.export_library("deploy_lib.so", tvm.contrib.ndk.create_shared)
    
    print("Exporting deploy_param.params parameters file to disk...")
    with open("deploy_param.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))
    
    print("Deploying to RPC!")
    tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
    key = "android"
    
    print("Connecting to {}:{}...".format(tracker_host, tracker_port))
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    
    print("Requesting session with key {} ...".format(key))
    # When running a heavy model, we should increase the `session_timeout`
    remote = tracker.request(key, priority=0, session_timeout=60)
    ctx = remote.cpu(0)
    
    print("Uploading...")
    remote.upload("deploy_lib.so")
    rlib = remote.load_module("deploy_lib.so")
    
    print("Running...")
    module = tvm.contrib.graph_runtime.GraphModule(rlib["default"](ctx))
    module.set_input(input_name, tvm.nd.array(input_data.numpy()))
    module.run()
    out0 = module.get_output(0)
    print(out0.shape)
    out1 = module.get_output(1)
    print(out1.shape)
    out2 = module.get_output(2)
    print(out2.shape)
    out3 = module.get_output(3)
    print(out3.shape)
    out4 = module.get_output(4)
    print(out4.shape)
    
    print("Benchmarking inference...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))
    
    print("Done!")

if __name__ == '__main__':
    init_seed(1234)
    main(parse_args())
