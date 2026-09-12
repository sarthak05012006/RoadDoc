"""Inspect-able CV fallback + session trend logic for Weather Whiplash."""
from __future__ import annotations
from collections import defaultdict
from io import BytesIO
from typing import Dict, List
import numpy as np
from PIL import Image, ImageFilter

LABELS = ["Dry", "Damp", "Drying", "Wet"]


def _channels(image: Image.Image, size=(240, 160)):
    im = image.convert("RGB").resize(size)
    arr = np.asarray(im).astype(np.float32)
    mx = arr.max(axis=2); mn = arr.min(axis=2)
    d = mx - mn
    s = np.where(mx == 0, 0, d / mx * 255)
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    return mx, s, gray


def _pseudo_probabilities(wetness: float, patchiness: float) -> dict:
    """Deterministic, signal-derived stand-in for class probabilities when running
    the heuristic fallback (no trained softmax available). Not random: built from
    the same wetness/patchiness scores the label itself is derived from."""
    dry = max(0.0, 1 - wetness/0.22) if wetness < 0.22 else max(0.0, 1 - (wetness-0.22)/0.35)
    wet = max(0.0, (wetness-0.60)/0.40) if wetness > 0.60 else 0.0
    mid = max(0.0, 1 - dry - wet)
    drying = mid * min(patchiness/0.6, 1.0)
    damp = mid - drying
    total = dry + damp + drying + wet
    if total <= 0:
        return {'Dry': 0.25, 'Damp': 0.25, 'Drying': 0.25, 'Wet': 0.25}
    return {'Dry': round(dry/total, 3), 'Damp': round(damp/total, 3),
            'Drying': round(drying/total, 3), 'Wet': round(wet/total, 3)}


def analyze_image(image: Image.Image) -> dict:
    v, s, gray = _channels(image)
    # Small Gaussian blur is a practical denoise equivalent for the original heuristic.
    vb = np.asarray(Image.fromarray(np.clip(v,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))).astype(np.float32)
    vmin, vmax, vmed = float(vb.min()), float(vb.max()), float(np.median(vb))
    thresh = max(vmin + 0.55*(vmax-vmin), vmed + 25)
    glare = (vb > thresh) & (s < 70)
    reflection = float(glare.mean())
    y0 = vb.shape[0]//2
    vv = vb[y0:]; ss = s[y0:]
    darkness = 1.0 - float(vv.mean())/255.0
    saturation = float(ss.mean())/255.0
    # Sobel-like gradients via finite differences.
    gx = np.diff(gray[y0:], axis=1, prepend=gray[y0:, :1])
    gy = np.diff(gray[y0:], axis=0, prepend=gray[y0:y0+1, :])
    edge_density = float((np.sqrt(gx*gx+gy*gy) > 20).mean())
    texture_norm = min(edge_density/0.15, 1.0)
    wetness = 0.55*min(reflection*6.0,1.0) + 0.25*(1-texture_norm) + 0.12*darkness + 0.08*saturation
    wetness = float(np.clip(wetness,0,1))

    scores=[]; grid=4; h,w=vb.shape
    ph,pw=h//grid,w//grid
    for gy_i in range(grid):
        for gx_i in range(grid):
            p_v=vb[gy_i*ph:(gy_i+1)*ph, gx_i*pw:(gx_i+1)*pw]
            p_s=s[gy_i*ph:(gy_i+1)*ph, gx_i*pw:(gx_i+1)*pw]
            p_glare=((p_v>thresh)&(p_s<70)).mean()
            p_dark=1-float(p_v.mean())/255
            scores.append(min(0.7*p_glare*6,1)*0.6+p_dark*0.4)
    patchiness=float(np.clip(np.sqrt(np.var(scores))*3,0,1))
    if wetness < 0.22: label='Dry'
    elif wetness >= 0.60: label='Wet'
    elif patchiness > 0.35: label='Drying'
    else: label='Damp'
    return {
        'label':label,
        'wetness_score':round(wetness,3),
        'patchiness_score':round(patchiness,3),
        'reflection_ratio':round(reflection,4),
        'texture_score':round(edge_density,4),
        'probabilities':_pseudo_probabilities(wetness, patchiness),
        'engine':'heuristic-cv-fallback',
    }


def trend_for(history: List[dict]) -> tuple[str, float|None]:
    recent=[float(x['wetness_score']) for x in history[-8:]]
    if len(recent)<2: return 'Not enough data yet', None
    x=np.arange(len(recent),dtype=float); y=np.asarray(recent)
    slope=float(np.polyfit(x,y,1)[0]) if len(recent)>1 else 0
    if slope>0.02: return 'Getting wetter', slope
    if slope<-0.02: return 'Drying out', slope
    return 'Stable', slope


def suggestion_for(label, trend):
    if label=='Wet' and trend!='Drying out': return 'Track is wet. Stay on full wet tyres.'
    if label=='Wet': return 'Track is wet but drying. Hold wets for now, watch the next laps closely.'
    if label=='Drying': return 'Track drying: tyre change window approaching — get intermediates ready.'
    if label=='Damp' and trend=='Drying out': return 'Damp and drying fast. Consider switching to intermediates soon.'
    if label=='Damp': return 'Track is damp. Intermediates recommended, monitor closely.'
    if label=='Dry' and trend=='Getting wetter': return 'Still dry but conditions worsening — be ready to react quickly.'
    if label=='Dry': return 'Track is dry and stable. No tyre change needed.'
    return 'Monitoring conditions.'


class TrackConditionTracker:
    def __init__(self): self.sessions=defaultdict(list)
    def analyze(self, image, session='default', model_result=None):
        result=model_result or analyze_image(image)
        hist=self.sessions[session]
        hist.append({'label':result['label'],'wetness_score':result['wetness_score']})
        trend,slope=trend_for(hist)
        result.update({'trend':trend,'slope':slope,'suggestion':suggestion_for(result['label'],trend),'history':hist[-12:]})
        return result
    def reset(self, session='default'): self.sessions.pop(session,None)
