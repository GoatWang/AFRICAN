# Experiments
## VideoCLIP (X: DONE, D: Doing, S:Suspend, C:Cancel)
- [X] Test VideoCLIP baseline
- [X] Find the best loss function (BCE, Focal, EQL)
    - [X] Test BCE on lr0.00015
    - [X] Test Focal on lr0.00015
    - [X] Test EQL on lr0.00015
- [X] prove bs128 > bs008 (lr: 0.0001)
- [D] check rand sampling works as well
- [C] variable bs (0008 -> 1024) vs 1024
- [C] verify segment in promp training performace
    - [C] using segment
        - [C] using BCE
        - [C] using Focal    
    - [C] using rare, uncommon, common(general)

## AFRICAN
- [X] test clip_cosine_infoNCE_8_uniform_augmix (lr=0.0001) (epoch=300)
- [X] test init_cosine_infoNCE_8_uniform_augmix (lr=0.0001) (epoch=300)
- [X] test clip_cosine_infoNCE_8_rand_augmix_000040 (epoch=300)
- [X] test clip_nodecay_infoNCE_8_rand_augmix_000030 (epoch=300)
- [ ] test clip_cosine_infoNCE_8_rand_augmix_000030 (epoch=100)

## AFRICAn Slowfast
- [X] AFRICA_nodecay_128_00015: test clip_cosine_infoNCE_8_uniform_augmix_000040_epoch52.ckpt on africa
- [X] test ViT-L-14.pt on africa
- [C] test with africa only without VideoCLIP
- [X] test clip_no_decay_infoNCE_8_rand_augmix_000030_epochxx.ckpt on africa
- [X] test cosine (lr=0.00015) (epoch=50)
- [X] test agmentation on FrameID representation: option to load video in dataset directly (not preprocessed representation) with smaller batch size

# paper
1. lrdecay solve log-tail: https://arxiv.org/pdf/2203.14197.pdf
2. adamW: 
3. GELU: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L126

# TODO:
- mv log to s3
- Write training script for variable bs
- Ask InternVideo how to combine VideoCLIP and VideoMAE
- Ask AnimalKingdom how to calculate the count of each segment (head, middle and tail): 
    > When constructing our animal action recognition dataset, we follow the work of [82] and divide the distri- bution into three different segments based on the number of samples in each action class. Specifically, we group all the 140 action classes in our dataset into the head segment (17 action classes that have more than 500 samples each), the middle segment (29 action classes that have 100 to 500 samples each), and the tail segment (94 action classes that have fewer than 100 samples each).
- write scripts for africa training with augmented input (not preprocessed img representation)

# DONE
- deal with text embedding using cuda
- batch_size into config
- Add (M)AP as metrics
- device into config
- Change version to datetime
- change ckpt dir to with model name and version
- sort metrics into collections
- unbalance training (focal loss (panelty on False Positive)) 
    - Use BCELoss (sigmoid + cross entropy) is better than applying sigmoid and use crossentropy: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
    - AnimalKingdom uses its own loss: data/AnimalKingdom/action_recognition/code/code_new/slowfast/slowfast/models/losses.py
    - different weight for different class (more weight for not frequent class)
- Think about the similarity of each class's prompt (visualiz its distribution)
    ```python
    import cv2
    import numpy as np
    import pandas as pd
    from config import config
    from matplotlib import pyplot as plt
    from Dataset import AnimalKingdomDataset

    sim_thres = 0.9
    X = np.load("/Users/jeremywang/BristolCourses/Dissertation/AnimalKingdomCLIP/Train/temp/text_features.npy")
    simmat = np.dot(X, X.T)

    # simmat_hist = simmat[np.triu(np.ones_like(simmat, dtype=bool), k=1)]
    # plt.hist(simmat_hist)
    # plt.show()

    _config = config()
    dataset = AnimalKingdomDataset(_config, split="train")
    df_action = dataset.df_action

    simmat[np.eye(simmat.shape[0], dtype=bool)] = 0
    sim_terms = set([tuple(sorted(idxs)) for idxs in np.stack(np.where(simmat > sim_thres)).T])
    df_sim_terms = pd.DataFrame(list(sim_terms), columns=['i0', 'i1'])
    df_sim_terms = pd.merge(df_sim_terms, df_action[['action', 'action_category', 'segment', 'count']], left_on='i0', right_index=True)
    df_sim_terms = pd.merge(df_sim_terms, df_action[['action', 'action_category', 'segment', 'count']], left_on='i1', right_index=True, suffixes=['_0', '_1'])
    df_sim_terms['score'] = df_sim_terms.apply(lambda row: simmat[row['i0'], row['i1']], axis=1)
    columns = ["i0", "i1", "score", "action_0", "action_1", "action_category_0", "action_category_1", "segment_0", "segment_1", "count_0", "count_1"]
    df_sim_terms.sort_values('score', ascending=False)[columns]
    ```
- Test the scheduler
    ```python
    import pandas as pd
    import numpy as np
    from transformers import get_cosine_schedule_with_warmup
    import torch

    x = [torch.autograd.Variable(torch.randn(10, 140).type(torch.float32), requires_grad=True)]
    optimizer = torch.optim.Adam(x, lr=0.0001)

    scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=10000,
                    num_training_steps=100000,
                )

    lr_list = []
    for i in range(100000):
        optimizer.step()
        lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    df = pd.DataFrame({
        'step': np.arange(100000),
        'lr': lr_list,
    })

    print(df)
    ```
- do research on the evaluation
- learn the different loss function
    - LDAM: panelize the loss of target dimension of 
        ```python
        # m_list: 1 / np.sqrt(np.sqrt(Nc)): minority get higher score
        batch_m = torch.matmul(self.m_list[None, :], targets.transpose(0, 1))
        # batch_m: 1 / np.sqrt(np.sqrt(Nc)): minority get higher score
        batch_m = batch_m.view((-1, 1))
        inputs_m = inputs - batch_m
        # inputs_m: minority get lower result

        output = torch.where(targets.type(torch.bool), inputs_m, inputs) # TODO: should be tested
        # keep inputs_m only on target dimension
        # pred on target dimension which is minority class will get lower result
        # loss on target dimension which is minority class will get larger loss
        ```

        ```python
        import numpy as np
        from config import config
        from Dataset import AnimalKingdomDataset

        _config = config()
        dataset = AnimalKingdomDataset(_config, split="train")
        df_action = dataset.df_action

        m_list = 1.0 / np.sqrt(np.sqrt(df_action['count'].tolist()))
        m_list = m_list * (max_m / np.max(m_list))

        fd_freq = df_action[['action', 'count']].copy()
        fd_freq['weight'] = m_list
        print(fd_freq)
        ```
    - EQL: 
        - Do not add any loss on minority class dimension if it is not target
        - Do not panelize FP of rare class. 
        ```python
        eql_w = 1 - self.beta_func() * self.threshold_func() * (1 - targets)
        # beta: just random, not important
        # threshold_func: count samller than lambda, meaning the minority class
        # (1 - targets): not be target
        # => if (minority class & not target): # do not add any loss on minority class dimension if it is not target
        #        eql_w = 0
        #    else:
        #        eql_w = 1
        loss = F.binary_cross_entropy_with_logits(self.inputs, targets, reduction=self.reduction, weight=eql_w)
        ```
- implement segment specific map metric log
- reimplement the map metrics logging for each prompt
- specify the model version using 'version' config
- check is the batch_size (32 -> 128) or the learning rate (0.0001 -> 0.00015)cause the poor performance of the model training. => batch_size count (32 better)
- ask for BlueCrystol
- sort code, remove notes
- change to use wandb
- try Focal Loss with fixed lr & the training result of Cosine Learning Rate seems not work
- Monitor mAP on validation result not training result
- saving ckpt filename metric (head, middle, tail) name error
- add in MetricCollection.clone(prefix='train_'), remember to also modify the moniter name
- write the first model to WandB
- Change the name of training_test_size to function_test_size
- move ckpts to s3
- add mAP calculation for head, middle and tail classes
- Combine the contrastive learning and VideoCLIP result 
- check the sampling method acceptable 
- test vision train_laryers (A100)
- contrastive learn on sequence order of the frames (survey & model building)

## Suspended
- test rand sampling of video frames
- PR to InternVideo on VideoReader (File Reader: )
- start index of Dataset
- use BlueCrystol



## Email
Hi Otto and Majid, 

After the first experiment, I wanted to provide you with an update on the current status and our future plan.

1. current status:
    1. developed the most basic model using VideoCLIP.
    2. completed the first experiment, training VideoCLIP with only the weight of the projection layer updated.
    3. The current mAP is 24.4% using cross-entropy loss (the State-of-the-Art 30% according to the original dataset paper).
    4. Each experiment takes about 110 hours to run on an A100 GPU, with 150 epochs (approximately 44 minutes per epoch). The mAP tends to converge around 150 epochs.

2. future plan:
    1. (Takes around 2 days) I am currently adjusting some hyper parameters such as learning rate and batch size, as I have observed that they significantly impact the training performance.
    2. (Takes around a week) I will explore different loss functions for the long-tail unbalanced dataset, such as Focal Loss, EQL, and LDAM as the paper suggested.
    3. (Takes around a week) Once I have determined the best hyperparameters and loss function, I will train a complete VideoCLIP model (not just the projection layers).
    4. (Takes around a week) Add other feature extractor, here two options:
        - Option1: Following the approach proposed in the original paper, I will add an animal classifier to distinguish actions performed by different animals.
        - Option2: I will explore using an unsupervised learning model, such as VideoMAE or contrastive learning, as an additional feature extractor.

If possible, could I have access to BlueCrystal to accelerate the experiment speed?
Please let me know if you have any questions or if there are any changes you'd like to make.
    


    
