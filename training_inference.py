from pathlib import Path
import re, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from fastai.vision.all import *
from PIL import Image

root = Path("UCLouvain RGB canopy dataset")
site = "participatif2020"
camera = "192.168.42.19"
mask_root = Path("/path/to/masks_root")

codes = ['background','bean','fennel','kale','cabbage']
n_classes = len(codes)

letter_to_class = {
    "A": "bean",
    "B": "fennel",
    "C": "cabbage",
    "D": "kale",
}

class_to_idx = {c:i for i,c in enumerate(codes)}

def is_img(p): return p.is_file() and p.suffix.lower() in {".jpg",".png"}
def is_mask(p): return p.is_file() and p.suffix.lower() in {".png",".jpg"}

def extract_code4(p):
    m = re.search(r'([A-Za-z]{4})(?=\.[^.]+$)', p.name)
    return m.group(1).upper() if m else None

def is_pure_code(code4):
    if not code4 or len(code4) != 4: return False
    if len(set(code4)) == 1: return True
    return (code4[0] == code4[2]) and (code4[1] == code4[3])

def pure_class_from_code(code4):
    if not is_pure_code(code4): return None
    letters = sorted(set(code4))
    if any(l not in letter_to_class for l in letters): return None
    cls = {letter_to_class[l] for l in letters}
    if len(cls) != 1: return None
    cls = next(iter(cls))
    return cls if cls in class_to_idx else None

def camera_files(root, site, camera):
    base = root/site
    return sorted([p for p in base.glob(f"*/{camera}/*") if is_img(p)])

def mask_path(img_path):
    rel = img_path.relative_to(root)
    p = mask_root/rel
    return p.with_suffix(".png")

def load_binary_mask(fn):
    m = np.array(Image.open(fn))
    if m.ndim == 3: m = m[...,0]
    m = (m > 0).astype(np.uint8)
    return torch.from_numpy(m)

def onehot_from_binary(mask01, fg_idx):
    bg = (mask01 == 0).float()
    fg = (mask01 == 1).float()
    out = torch.zeros((n_classes, *mask01.shape), dtype=torch.float32)
    out[0] = bg
    out[fg_idx] = fg
    return out

class MultiChannelMask(Transform):
    def encodes(self, x:Path):
        mfn = mask_path(x)
        if not mfn.exists(): raise FileNotFoundError(str(mfn))
        code4 = extract_code4(x)
        cls = pure_class_from_code(code4)
        if cls is None: raise ValueError(f"Non-pure or unmapped code in filename: {x.name}")
        fg_idx = class_to_idx[cls]
        m01 = load_binary_mask(mfn)
        return TensorImage(onehot_from_binary(m01, fg_idx))

fnames = camera_files(root, site, camera)
fnames = [p for p in fnames if mask_path(p).exists() and (pure_class_from_code(extract_code4(p)) is not None)]
if len(fnames) == 0: raise RuntimeError("No usable image/mask pairs found")

item_tfms = [Resize(512, method='pad', pad_mode='zeros')]
batch_tfms = [*aug_transforms(size=512, flip_vert=True, max_rotate=10, max_zoom=1.1, max_warp=0.1, max_lighting=0.2), Normalize.from_stats(*imagenet_stats)]

dblock = DataBlock(
    blocks=(ImageBlock, TransformBlock(type_tfms=MultiChannelMask())),
    get_items=noop,
    get_y=noop,
    splitter=RandomSplitter(0.2, seed=42),
    item_tfms=item_tfms,
    batch_tfms=batch_tfms
)

dls = dblock.dataloaders(fnames, bs=8, num_workers=0)

class FocalLossOneHot(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, eps=1e-8):
        super().__init__()
        self.gamma, self.eps = gamma, eps
        self.alpha = alpha

    def forward(self, logits, targ):
        p = logits.softmax(1).clamp(self.eps, 1 - self.eps)
        pt = (p * targ).sum(1)
        w = (1 - pt).pow(self.gamma)
        if self.alpha is not None:
            a = torch.as_tensor(self.alpha, device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
            at = (a * targ).sum(1)
            w = w * at
        return (-w * pt.log()).mean()

class SoftDiceOneHot(nn.Module):
    def __init__(self, smooth=1e-6, include_bg=True):
        super().__init__()
        self.smooth, self.include_bg = smooth, include_bg

    def forward(self, logits, targ):
        p = logits.softmax(1)
        if not self.include_bg:
            p = p[:,1:]
            targ = targ[:,1:]
        dims = (0,2,3)
        i = (p * targ).sum(dims)
        u = p.sum(dims) + targ.sum(dims)
        d = (2*i + self.smooth) / (u + self.smooth)
        return 1 - d.mean()

class LogCoshDiceOneHot(nn.Module):
    def __init__(self, include_bg=True):
        super().__init__()
        self.d = SoftDiceOneHot(include_bg=include_bg)
    def forward(self, logits, targ):
        x = self.d(logits, targ)
        return torch.log(torch.cosh(x + 1e-12))

class ComboLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, include_bg=True):
        super().__init__()
        self.f = FocalLossOneHot(gamma=gamma, alpha=alpha)
        self.d = LogCoshDiceOneHot(include_bg=include_bg)
    def forward(self, logits, targ):
        return self.f(logits, targ) + self.d(logits, targ)

def dice_multi_onehot(inp, targ):
    p = inp.argmax(1)
    y = targ.argmax(1)
    num = 0.0
    den = 0.0
    for c in range(n_classes):
        pc = (p == c)
        yc = (y == c)
        inter = (pc & yc).float().sum()
        union = pc.float().sum() + yc.float().sum()
        num += (2*inter)
        den += union
    return (num / (den + 1e-8)).item()

class DiceMultiOneHot(Metric):
    def reset(self): self.vals=[]
    def accumulate(self, learn):
        self.vals.append(dice_multi_onehot(learn.pred.detach(), learn.y.detach()))
    @property
    def value(self): return float(np.mean(self.vals)) if self.vals else None
    @property
    def name(self): return "dice_multi"

def foreground_acc_onehot(inp, targ):
    p = inp.argmax(1)
    y = targ.argmax(1)
    return ((p != 0) == (y != 0)).float().mean()

learn = unet_learner(
    dls,
    resnet50,
    n_out=n_classes,
    loss_func=ComboLoss(gamma=2.0, alpha=None, include_bg=True),
    metrics=[DiceMultiOneHot(), AccumMetric(foreground_acc_onehot, flatten=False)],
    pretrained=True
)

learn.fit_one_cycle(30, lr_max=3e-3, moms=(0.95, 0.85))
learn.save(f"fcnn_resnet50_{camera.replace('.','_')}_binary_to_multich")

out_dir = Path("output_masks"); out_dir.mkdir(parents=True, exist_ok=True)
test_root = Path("multispecies_images_root")

test_files = sorted([
    p for p in test_root.glob(f"*/{camera}/*")
    if is_img(p)
])

dl = learn.dls.test_dl(test_files, with_labels=False)
logits = learn.get_preds(dl=dl)[0]
pred_ids = logits.argmax(1).cpu().numpy()

for f, m in zip(test_files, pred_ids):
    out = out_dir/f"{f.parent.name}_{f.stem}_pred.png"
    Image.fromarray(m.astype("uint8")).save(out)

