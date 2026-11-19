"""Measure RT-DETRv2 deploy latency: fp32 vs fp16 autocast, best of several passes.

Contention only ever adds time, so the minimum of per-pass medians is the fair
estimate.  Used to compare against the method-3 author's reported numbers.
"""
import os, sys, time, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "method3"))

import torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from src.core import YAMLConfig

CFGS = [
    ("default 640", "method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
     "method3/output/default_params.pth"),
    ("stride4 config", "method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_stride4.yml",
     "method3/output/stride4.pth"),
]
IMGS = sorted(os.listdir("common/dataset/images/test"))[:24]


def build(cfg_path, ckpt):
    cfg = YAMLConfig(cfg_path, resume=ckpt)
    chk = torch.load(ckpt, map_location="cpu")
    state = dict(chk["ema"]["module"] if "ema" in chk else chk["model"])
    ms = cfg.model.state_dict()
    for k in ("decoder.anchors", "decoder.valid_mask"):
        if k in state and k in ms and state[k].shape != ms[k].shape:
            state.pop(k)
    cfg.model.load_state_dict(state, strict=False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = cfg.model.deploy()
            self.p = cfg.postprocessor.deploy()

        def forward(self, x, s):
            return self.p(self.m(x), s)

    return M().to("cuda").eval(), tuple(cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))


def bench(model, size, amp, passes=3):
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    per_pass = []
    with torch.no_grad():
        for _ in range(passes):
            ts = []
            for f in IMGS:
                im = Image.open(os.path.join("common/dataset/images/test", f)).convert("RGB")
                w, h = im.size
                x = tf(im)[None].cuda()
                o = torch.tensor([w, h])[None].cuda()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        model(x, o)
                else:
                    model(x, o)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            per_pass.append(statistics.median(ts))
    return per_pass


for label, cfg, ckpt in CFGS:
    model, size = build(cfg, ckpt)
    for amp in (False, True):
        p = bench(model, size, amp)
        print("%-12s amp=%-5s pass medians %s -> best %.1f ms/img"
              % (label, amp, ["%.1f" % v for v in p], min(p)), flush=True)
    del model
    torch.cuda.empty_cache()
