"""Evaluate the saved model and print a confusion matrix + classification report."""
from pathlib import Path
import json, torch
from torch.utils.data import DataLoader
from torchvision import datasets,models,transforms
from sklearn.metrics import classification_report,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'models/weather_whiplash_model'; DATA=ROOT/'dataset'
classes=json.loads((OUT/'classes.json').read_text()); ckpt=torch.load(OUT/'model.pth',map_location='cpu')
model=models.resnet18(weights=None); model.fc=torch.nn.Linear(model.fc.in_features,len(classes)); model.load_state_dict(ckpt['model_state']); model.eval()
tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
ds=datasets.ImageFolder(DATA/'val',transform=tf); dl=DataLoader(ds,batch_size=32,shuffle=False)
yt=[]; yp=[]
with torch.no_grad():
    for x,y in dl: yt.extend(y.tolist()); yp.extend(model(x).argmax(1).tolist())
print(classification_report(yt,yp,target_names=classes,digits=4)); print('Confusion matrix:\n',confusion_matrix(yt,yp))
