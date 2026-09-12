"""Optional trained classifier service. Uses a fine-tuned ResNet18 checkpoint.

torch/torchvision are optional at runtime: if they aren't installed (or no
checkpoint has been trained yet), the service simply reports itself as
unavailable and the app falls back to the heuristic classifier — the same
graceful-degradation approach used for the CV pipeline elsewhere in this
project, rather than the whole backend failing to start.
"""
from __future__ import annotations
from pathlib import Path
import json

try:
    import torch
    from torchvision import models, transforms
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'models/weather_whiplash_model'
WEIGHTS=MODEL_DIR/'model.pth'
CLASSES=MODEL_DIR/'classes.json'

class WeatherModel:
    def __init__(self):
        self.model=None; self.classes=None
        if not _TORCH_AVAILABLE:
            print('[model] torch/torchvision not installed — running on heuristic CV only.')
            return
        self.device=torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.load()
    @property
    def available(self): return self.model is not None
    def load(self):
        if not (_TORCH_AVAILABLE and WEIGHTS.exists() and CLASSES.exists()): return
        try:
            self.classes=json.loads(CLASSES.read_text())
            m=models.resnet18(weights=None)
            m.fc=torch.nn.Linear(m.fc.in_features,len(self.classes))
            ckpt=torch.load(WEIGHTS,map_location='cpu')
            m.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
            self.model=m.to(self.device).eval()
        except Exception as e:
            print(f'[model] checkpoint could not be loaded: {e}')
            self.model=None
    def predict(self,image:Image.Image):
        if not self.available: return None
        x=self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs=torch.softmax(self.model(x),dim=1)[0]
        idx=int(torch.argmax(probs)); conf=float(probs[idx])
        label=self.classes[idx].capitalize()
        probabilities={self.classes[i].capitalize():round(float(p),4) for i,p in enumerate(probs.tolist())}
        # Keep a wetness-like scalar for the existing trend UI.
        wetness_map={'Dry':0.05,'Damp':0.30,'Drying':0.50,'Wet':0.84}
        return {'label':label,'confidence':round(conf,4),'wetness_score':wetness_map.get(label,0.5),'patchiness_score':0.0,'probabilities':probabilities,'engine':'resnet18-weather-whiplash'}
