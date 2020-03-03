KDD2020: DADG
Source code of: Discriminative Adversarial Domain Generalization with Meta-learning based Cross-domain Validation

Reproduce the paper:
Preparation:
1. Get data:
Please download the folder "data" from: https://drive.google.com/open?id=1CpXITj6wgkCkf3P-fgi2c-1IAosM06sU, and store the folder at the same path of code.

2. Please download the alexnet model from: and store the folder at the same path of code.
Create two empty folders: "Best_models" and "netD", for store the models and Discriminators.

To reproduce the result of PACS on AlexNet:
Run "run_PACS.sh" to reproduce the result (target domain: cartoon) of PACS on AlexNet.

To reproduce the result of other domains, change the command in "run_PACS.sh" to:
python run_idea.py --source _ _ _ --target _

To reproduce the result on ResNet18, change the command in "run_PACS.sh" to:
python run_idea.py --source _ _ _ --target _ --iteration 1000 --model_name resnet18

Add the domain name on "_". (P/A/C/S)
