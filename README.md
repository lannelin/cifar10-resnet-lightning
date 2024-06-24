# cifar10-resnet-lightning
Starter project for lightning CLI training of CIFAR-10 with ResNet18

Current configuration achieves 86.4% accuracy on validation set after 30 epochs.

```bash
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

dev:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Pretrained

A model trained for 30 epochs, achieving 86.4% acc on val, can be found in safetensors format in [releases](https://github.com/lannelin/cifar10-resnet-lightning/releases). See below instructions for conversion instructions back to torch format .ckpt.

## Run

take a look at the config in `configs/training.yaml` before running and update any values necessary (particularly necessary for wandb logging section).

```bash
   python runner.py fit -c configs/training.yaml \
 --trainer.logger.name cifar_resnet \
  --trainer.logger.save_dir weights/cifar_resnet
```

you can specify args to replace keys within the config, as shown here with `trainer.logger.name` and `trainer.logger.save_dir`

## To/From SafeTensors


TODO:
- review security assumptions for this section:
  - currently pickle is constructed locally rather than downloading pickle of arbitrary content
  - does use of safetensors *sanitize* in any way? we're still converting back before load
- get optional deps working with local editable install

```
pip install "safetensors==0.4.3"
```

`convert_checkpoint.py` heavily borrows from https://github.com/huggingface/safetensors/blob/v0.4.3/bindings/python/convert.py

We can use this script to convert to and from [safetensor format](https://github.com/huggingface/safetensors). This is useful for sharing models without using pickle (unsafe, can run arbitrary code). **However, at the moment we're still converting back to a pickle seralized format, before loading locally**.

This conversion is lossy and will only maintain the state_dict.

```bash
python convert_checkpoint.py --help
```
