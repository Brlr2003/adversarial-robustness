"""Reliability of the PGD robust accuracy: re-run PGD-20 on the full 10,000-image
test set with 3 different random-start seeds, for both models, and report mean +/- std.
Directly answers the examiner note that the reliability of the results was missing."""
import os, sys, json, statistics, torch
from torchvision import datasets, transforms
sys.path.insert(0, os.path.abspath('.'))
from src.models.resnet import resnet18_cifar10
from src.attacks.pgd import pgd_attack
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def load(p):
    m=resnet18_cifar10(); ck=torch.load(p,map_location=dev,weights_only=True); m.load_state_dict(ck['model_state_dict']); return m.to(dev).eval()
tf=transforms.ToTensor(); test=datasets.CIFAR10('./data',train=False,download=True,transform=tf)
X=torch.stack([test[i][0] for i in range(len(test))]); Y=torch.tensor([test[i][1] for i in range(len(test))])
out={}
for tag,ck in [('standard','checkpoints/standard_best.pt'),('robust','checkpoints/robust_best.pt')]:
    m=load(ck); accs=[]
    for seed in range(3):
        torch.manual_seed(seed); correct=0
        for i in range(0,len(X),500):
            adv=pgd_attack(m, X[i:i+500], Y[i:i+500], 8/255, 2/255, 20, True, dev)
            with torch.no_grad(): correct += (m(adv.to(dev)).argmax(1).cpu()==Y[i:i+500]).sum().item()
        accs.append(correct/100.0); print(f"{tag} seed {seed}: PGD acc {accs[-1]:.2f}%", flush=True)
    out[tag]={'accs':accs,'mean':round(statistics.mean(accs),2),'std':round(statistics.pstdev(accs),3)}
    print(f"{tag}: mean {out[tag]['mean']:.2f} std {out[tag]['std']:.3f}", flush=True)
json.dump(out, open('results/pgd_reliability.json','w'), indent=2); print("saved results/pgd_reliability.json")
