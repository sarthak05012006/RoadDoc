"""Predict one local image with the trained checkpoint."""
from pathlib import Path
import argparse,json,torch
from PIL import Image
from torchvision import models,transforms
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'models/weather_whiplash_model'
p=argparse.ArgumentParser(); p.add_argument('image'); args=p.parse_args()
classes=json.loads((OUT/'classes.json').read_text()); m=models.resnet18(weights=None); m.fc=torch.nn.Linear(m.fc.in_features,len(classes)); m.load_state_dict(torch.load(OUT/'model.pth',map_location='cpu')['model_state']); m.eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
x=tf(Image.open(args.image).convert('RGB')).unsqueeze(0)
with torch.no_grad(): p=torch.softmax(m(x),1)[0]
i=int(p.argmax()); print({'label':classes[i],'confidence':round(float(p[i]),4),'probabilities':{c:round(float(p[j]),4) for j,c in enumerate(classes)}})
