# TODO:
1. unbalance training (focal loss (panelty on False Positive)) 
    - Use BCELoss (sigmoid + cross entropy) is better than applying sigmoid and use crossentropy: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    - AnimalKingdom uses its own loss: data/AnimalKingdom/action_recognition/code/code_new/slowfast/slowfast/models/losses.py
    - different weight for different class (more weight for not frequent class)
2. Think about the similarity of each class's prompt (visualiz its distribution)

# Next Steps
5. do research on the evaluation
2. change to use wandb
1. test vision train_laryers (A100)

# DONE
- deal with text embedding using cuda
- batch_size into config
2. Add (M)AP as metrics
- device into config
1. Change version to datetime
3. change ckpt dir to with model name and version
3. sort metrics into collections

## Suspended
4. test rand sampling of video frames
- PR to InternVideo on VideoReader
- start index of Dataset

