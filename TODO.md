# TODO:
- check is the batch_size (32 -> 128) or the learning rate (0.0001 -> 0.00015)cause the poor performance of the model training.
- implement segment specific map metric log
- ask for BlueCrystol

# Next Steps
- test vision train_laryers (A100)
- contrastive learn on sequence order of the frames (survey & model building)

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

## Suspended
- test rand sampling of video frames
- PR to InternVideo on VideoReader
- start index of Dataset
- change to use wandb

