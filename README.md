Whisper fine-tuning always been a black-magic since we don't have enough proper documentation on **how to fine-tune the model and structure the dataset, especially at scale**. In this repo, I provide both the code for training and preprocessing an existing dataset; And train on it. 

I have trained Whisper Medium on Hebrew dataset. I have trained it on 80,000 training files and 10,000 validation files. [I also provide the dataset alongside this fine-tune training code]

#### How to train?
1. Download the dataset from HF and store in working dir. [https://huggingface.co/datasets/sleeping-ai/hebrew-whisper]

2. Clone this repo in working repo and have the files in working repo. and hit train.py with
```bash
torchrun --nproc-per-node=8 train.py
```

#### Barebone training details
Multi-GPU training on 8xH100s (80GB) for two hours.

#### Acknowledgement
A big thanks to [TensorPool](https://tensorpool.dev/) for supporting this.  
