# AI-project5

多模态情感分类



### 项目依赖

在python3.8下实现，运行前需要安装下述依赖

```shell
transformers==4.19.2
pandas==1.4.3
numpy==1.21.2
tqdm==4.64.0
scikit-learn==1.0.1
matplotlib==3.4.3
argparse==1.4.0
torch==1.10.2
```

可以直接运行`pip install -r requirements.txt`安装所有依赖



### 项目结构

```shell
.
├── dataset # 数据集
│   ├── data # 包括所有的训练文本和图片，每个文件按照唯一的guid命名
│   ├── test_without_label.txt # 数据的guid和空的情感标签
│   └── train.txt # 数据的guid和对应的情感标签
├── draw.py # 训练之后对loss和acc作图
├── logs # 保存训练的loss和acc
│   ├── decision_fusion
│   │   └── metrics.csv
│   └── feature_fusion
│       └── metrics.csv
├── model # 模型的实现
│   ├── bert.py
│   ├── mutil_model.py
│   └── vgg.py
├── pictures # 训练之后对loss和acc的图片
│   ├── compare.png
│   ├── decision_fusion.png
│   └── feature_fusion.png
├── pretrained-model # 加载预训练模型的缓存，第一次运行后自动生成
│   └── bert-base-uncased
│       ├── model
│       │   ├── config.json
│       │   └── pytorch_model.bin
│       └── tokenizer
│           ├── special_tokens_map.json
│           ├── tokenizer_config.json
│           └── vocab.txt
├── result.txt # 测试数据结果文件
├── run.py # 训练和推断
├── trained_model # 训练好的模型参数保存
│   ├── decision_fusion
│   │   └── model.pth
│   └── feature_fusion
│       └── model.pth
└── utils # 工具的实现
    ├── data.py # 数据处理
    ├── draw.py # 画图
    ├── inference.py # 对测试作推断
    └── train.py # 训练
```



### 运行说明

#### 参数说明

```
usage: run.py [-h] [--train] [--fusion_level {feature,decision}] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS] [--gpu]
              [--save_model]

choose fusion level and hyper parameters for mutil model classification

optional arguments:
  -h, --help            show this help message and exit
  --train               train and then infer or infer directly
  --fusion_level {feature,decision}
                        choose model fusion level, feature or decision
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        choose learning rate
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        choose batch size
  --num_epochs NUM_EPOCHS, -n NUM_EPOCHS
                        choose number of epochs
  --gpu, -g             use gpu or not
  --save_model          save model parameters for inference or not

```

#### 运行实例

训练并生成测试结果

```shell
python run.py --train --fusion_level feature  --gpu --save
```
```shell
python run.py --train --fusion_level decision  --gpu --save
```

如果有训练好的模型保存在trained_model中，可直接生成测试结果
```shell
python run.py --fusion_level feature --gpu
```
训练两种模型后，可以运行`draw.py`做图查看模型训练过程
```shell
python draw.py
```