# Source code of: "Discriminative Adversarial Domain Generalization with Meta-learning based Cross-domain Validation"

Reproduce the paper:

1. Preparation:
  1.1 Get data:
    Please download the folder "data" from: https://drive.google.com/file/d/125Ks-0Au9VNa9LifeveV2wKvx4k4lw6D/view?usp=sharing, and store the folder at the same path of code.

  1.2 Load the pretrain AlexNet
    Please download the alexnet model from: https://drive.google.com/file/d/1WkDSnZYofOnK39-OW2oXZlJjmhlzXdKx/view?usp=sharing, and store the folder at the same path of code.

2. Create empty folder "Best_models" to store the models.

3. To reproduce the result of PACS on AlexNet:

  Run "run_PACS.sh" to reproduce the result (target domain: cartoon) of PACS on AlexNet.

4. To reproduce the result of other domains, change the command in "run_PACS.sh" to:

  python run_idea.py --source _ _ _ --target _

5. To reproduce the result on ResNet18, change the command in "run_PACS.sh" to:

  python run_idea.py --source _ _ _ --target _ --model_name resnet18,

  where the "_" should the domain name (P/A/C/S)
