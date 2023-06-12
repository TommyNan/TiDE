# TiDE
A unofficial pytorch implementation of "Long-term Forecasting with TiDE: Time-series Dense Encoder" and its sample code of applications

Link to paper: [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://arxiv.org/pdf/2304.08424.pdf)

Official Code written by Tensorflow: [Code](https://github.com/google-research/google-research/tree/master/tide)

# Usage
1. Config model

   edit the ```Tide_{dataset_name}.yaml``` file in the dir of config/TiDE/
   
2. Train

   ```python main_TiDE.py --dataset 'dataset_name' ```
   
# Details
1. The dimension of **Attribute** is ```[num_nodes, num_hidden_attribute]```. Here ```num_nodes``` is equal to the number of nodes in the original dataset, i.e., 7 for the ETT dataset, 862 for the traffic dataset. ```num_hidden_attribute=16```.
2. Dynamic Covariates' dimension is equal to 7 or 25, depending on whether you use ```holiday features``` or not, we use ```freq: 'B'``` to represent dynamic covaraites without holiday features and ```freq: 'S'``` to represent dynamic covaraites with holiday features
