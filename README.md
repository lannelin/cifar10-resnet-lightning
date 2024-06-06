# cifar10-resnet-lightning
starter project for lightning CLI training of CIFAR-10 with ResNet18

Current configuration achieves 86.4% accuracy on validation set after 30 epochs.

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install lightning torch torchvision wandb git+https://github.com/lannelin/lightning-bolts.git@lightning2  'jsonargparse[signatures]>=4.27.7'
```

dev:
```bash
pip install precommit
```
