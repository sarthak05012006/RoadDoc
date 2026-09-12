"""Fine-tune ResNet18 on Weather Whiplash classes.

Mac: PyTorch will use MPS when available. A pretrained ImageNet backbone is
used by default; --no-pretrained avoids downloading weights.
"""
from pathlib import Path
import argparse, json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset'; OUT=ROOT/'models/weather_whiplash_model'; OUT.mkdir(parents=True,exist_ok=True)
CLASSES=['dry','damp','drying','wet']

p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--no-pretrained',action='store_true'); p.add_argument('--workers',type=int,default=0); args=p.parse_args()

device=torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
train_tf=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(5),transforms.ColorJitter(brightness=.15,contrast=.15,saturation=.12),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
val_tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
train_ds=datasets.ImageFolder(DATA/'train',transform=train_tf); val_ds=datasets.ImageFolder(DATA/'val',transform=val_tf)
train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers); val_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
print('classes:',train_ds.classes,'device:',device,'train:',len(train_ds),'val:',len(val_ds))
weights=None if args.no_pretrained else models.ResNet18_Weights.DEFAULT
try: model=models.resnet18(weights=weights)
except Exception as e:
    print('Pretrained weights unavailable, training from random initialization:',e); model=models.resnet18(weights=None)
model.fc=nn.Linear(model.fc.in_features,len(train_ds.classes)); model=model.to(device)
criterion=nn.CrossEntropyLoss(); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
best=0.0
for epoch in range(1,args.epochs+1):
    model.train(); total=correct=loss_sum=0
    for x,y in train_dl:
        x,y=x.to(device),y.to(device); opt.zero_grad(); out=model(x); loss=criterion(out,y); loss.backward(); opt.step(); loss_sum+=loss.item()*len(y); correct+=(out.argmax(1)==y).sum().item(); total+=len(y)
    model.eval(); vc=vt=0
    with torch.no_grad():
        for x,y in val_dl:
            out=model(x.to(device)); vc+=(out.argmax(1).cpu()==y).sum().item(); vt+=len(y)
    tr=correct/total; va=vc/vt if vt else 0
    print(f'epoch {epoch:02d}/{args.epochs} train_acc={tr:.4f} val_acc={va:.4f}')
    if va>=best:
        best=va; torch.save({'model_state':model.state_dict(),'classes':train_ds.classes,'val_accuracy':best},OUT/'model.pth')
        (OUT/'classes.json').write_text(json.dumps(train_ds.classes))
print(f'Best validation accuracy: {best:.4f}')
