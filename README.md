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

## Run

take a look at the config in `configs/training.yaml` before running and update any values necessary (particularly necessary for wandb logging section).

```bash
   python runner.py fit -c configs/training.yaml \
 --trainer.logger.name cifar_resnet \
  --trainer.logger.save_dir weights/cifar_resnet
```

you can specify args to replace keys within the config, as shown here with `trainer.logger.name` and `trainer.logger.save_dir`

## To Safetensors

`convert_checkpoint_to_safetensors.py` heavily borrows from https://github.com/huggingface/safetensors/blob/v0.4.3/bindings/python/convert.py

We can use this script to convert to [safetensor format](https://github.com/huggingface/safetensors). This is useful for sharing models without using pickle (unsafe, can run arbitrary code).

This conversion is lossy and will only maintain the state_dict.

```bash
python convert_checkpoint_to_safetensors.py --help
```

## Using pretrained model in Safetensor format

add path to file in config at `model.safetensors_path` or specify in command line using `--model.safetenors_path`. E.g.

```bash
python runner.py test -c configs/training.yaml \
  --model.safetensors_path /path/to/model.safetensors
```

A model trained for 30 epochs, achieving 86.4% acc on val, can be found in safetensors format in [releases](https://github.com/lannelin/cifar10-resnet-lightning/releases).
