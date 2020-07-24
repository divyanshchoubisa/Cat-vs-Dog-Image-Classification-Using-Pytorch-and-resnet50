# Cat-vs-Dog-Image-Classification-Using-Pytorch-and-resnet50
This project is in Pytorch. This project will show you cat and dog image classification using resnet50.The Dataset Used in this project was taken from kaggle.
The dataset used was from:
https://www.kaggle.com/tongpython/cat-and-dog

# Cat-Dog Classfication Problem
The Cat-Dog Classification Problem is a standard computer vision problem to make classification between cat and dog.
# What  is Transfer Learning 
It is a technique used in area of artifical intellegence in which we use an already developed model as the starting point for a similar related task. In computer vision it is an popular technique to use pre-trained models to solve some similar related tasks.<br/>
Some famous models are
  - AlexNet
  - VGG
  - ResNet
  - SqueezeNet
  - DenseNet
  - Inception v3
  - GoogLeNet
  - ShuffleNet v2
  - MobileNet v2
  - ResNeXt
  - Wide ResNet
  - MNASNet
  
# ResNet
For this project I have used ResNet50 model as a starting point. And then I modified final fc layer as follows: <br/>
``` python
     model.fc = nn.Sequential(
               nn.Linear(2048, 270),
               nn.ReLU(inplace=True),
               nn.Linear(270,90 ),
               nn.ReLU(inplace=True),
               nn.Linear(90,2)).cuda()
```
<br/>
ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. Actually I have used ResNet50 Design which is a convolutional neural network that is 50 layers deep. The pretrained network can help classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. And so this was an very good choice for me to use this pre-trained model for my Cat-Dog Image Classification Project.
# Loss Function
As my network's final layer was a linear layer and so applying a softmax function is good so that we can have a probability distribution for our output classes and so I  used CROSSENTROPYLOSS() as my loss function.Pytorch CROSSENTROPYLOSS() combines nn.LogSoftmax() and nn.NLLLoss() in one single class which was the reason I didnt apply nn.LogSoftmax() the fc final output layer because Pytorch's CROSSENTROPYLOSS() whould do that for me and calculate loss through nn.NLLLoss().[Check Pytorch Doc for more details](https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html) <br/>
[And here is the Training Loss curve](https://github.com/divyanshchoubisa/Cat-vs-Dog-Image-Classification-Using-Pytorch-and-resnet50/blob/master/Cat%20and%20Dog%20Classifier/Training%20Loss.png)




