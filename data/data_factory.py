from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
"""
Choose the mode and dataset
mode: ['train', 'val', 'test', 'pred']
dataset: ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ELEC', 'EXCHANGE', 'ILI', 'TRAFFIC', 'WEATHER']

"""

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ELEC': Dataset_Custom,
    'EXCHANGE': Dataset_Custom,
    'ILI': Dataset_Custom,
    'TRAFFIC': Dataset_Custom,
    'WEATHER': Dataset_Custom,
}


def data_provider(mode, **configs):
    """
    mode choose from ['train', 'val', 'test', 'pred']

    """
    data_configs = configs['data']
    model_configs = configs['model']

    Data = data_dict[data_configs['dataset_name']]
    # timeenc default: 1
    timeenc = 0 if model_configs['embed_type'] != 'timeF' else 1

    if mode == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = data_configs['batch_size']
        freq = data_configs['freq']
    elif mode == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = data_configs['freq']
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = data_configs['batch_size']
        freq = data_configs['freq']

    data_set = Data(dataset_filename=data_configs['dataset_filename'],
                    mode=mode,
                    in_label_out_lens=[model_configs['in_lens'], model_configs['label_lens'], model_configs['out_lens']],
                    task_type=data_configs['task_type'],
                    target=data_configs['target'],
                    scale=True,
                    timeenc=timeenc,
                    freq=freq
                    )

    print(mode, len(data_set))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=data_configs['num_workers'],
        drop_last=drop_last
    )

    return data_set, data_loader

