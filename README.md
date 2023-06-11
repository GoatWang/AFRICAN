# Weights Downloads
```
wget -P ./Train/weights/ https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
wget -P ./Train/weights/ https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/pretrain/InternVideo-MM-L-14.ckpt
```

# Fail Videos
- /Users/jeremywang/BristolCourses/Dissertation/data/AnimalKingdom/action_recognition/dataset/video/ZAKHHVKA.mp4

# test script 
```
python3 Model.py
python3 Dataset.py
python3 Transform.py
python3 VideoReader.py
```

# Train
```
# mac
python3 train.py

# colab
python3 train.py with 'device=cuda'
```