# NLP_LSTM_homework

## LSTM模型
本项目的内容是：自己完成搭建LSTM网络，不再调用nn.LSTM、nn.LSTMCell，而使用nn.Linear、nn.Parameter来搭建网络

## 文件介绍
LSTMLM（源代码）.py：使用Pytorch包装好的LSTM函数

LSTMLM.py：手动实现单层LSTM

DoubleLSTMLM.py：手动实现双层LSTM

## 公式原理
![image](https://github.com/Hygge321/NLP_LSTM_homework/blob/main/LSTM/%E5%85%AC%E5%BC%8F%E5%8E%9F%E7%90%86%E5%9B%BE%E7%89%87.png)

## 心得体会
在这次LSTM的实践中，就我个人而言实践的过程还是比较困难的，因为需要从“用轮子”转变到“造轮子”，需要去理解LSTM的模型和模型中的计算公式，在理解之后的读代码和实现代码上我也是花费了不少时间，时间主要用在理解数据预处理的方式和导入训练模型中数据的维度。然后如何才能和图片公式中的参数对应起来让我大费周折。不过方法总比困难多，就是因为实践过程中遇到了不少疑难，所以在完成实验后收获也一样颇丰。同时我也完成了拓展实现双层LSTM，这是我没想到的。毫不夸张地说，实践前我对LSTM的理论理解不是很多，通过对理论的实践才真正让我理解LSTM模型。
