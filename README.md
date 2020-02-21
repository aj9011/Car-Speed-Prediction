# speed_prediction
Speed prediction uses 3D resnet model for video feature extraction and applies regression to predict the speed of frames

[Video feature extraction repository](https://github.com/kenshohara/3D-ResNets-PyTorch)   

This 3D model is for video classification, then we modified it to apply it on speed regression as well as speed dataset (Comma2k19)

[Comma2k19 Dataset repository](https://github.com/commaai/comma2k19)  

# Installation

- This code uses pytorch for training and testing. you may need "conda install pytorch==1.2.0"
- For loading Comma2k19 dataset, you need to use openpilot tools(https://github.com/commaai/openpilot-tools). This tool will reduce loading data time and cpu computation cost comparing to cv2.VideoCapture()

# Training and testing 
```
CUDA_VISIBLE_DEVICES=5 python3 main_speed_node3.py
```

# Results

- For speed prediction on Comma2k19 using 3D resnet video feature extraction, the resnet18 model currently obtained the best performance. After 11 epochs, resnet18 achieved 1.5 (m/s) mae error. we show the training and testing procedure in the following table: 

![training_procedure](https://gitlab.com/agilesoda/speed_prediction/blob/master/results_comma2k19/training_procedure.PNG)

- So far there are no algorithms applied yet on comma2k19, for comparing, we may need to look on other datasets. The paper "[Learning to Steer by Mimicking Features from Heterogeneous Auxiliary Networks](https://arxiv.org/abs/1811.02759)" has shown that the best mae performance of udacity dataset and comma.ai dataset are 1.6(m/s) and  0.7(m/s), respectively.


This is leaderboard of "Learning to Steer by Mimicking Features from Heterogeneous Auxiliary Networks":

![performance_reference](https://gitlab.com/agilesoda/speed_prediction/blob/master/results_comma2k19/reference.PNG)


- These results show that the result of applying resnet18 on comma2k19 maybe not bad.  


 