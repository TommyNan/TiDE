import os.path
import numpy as np
from .models import TiDE
import torch
import torch.nn as nn
import torch.optim
from data.data_factory import data_provider
from lib.metrics import metric
from lib.util import visual, EarlyStopping, test_params_flop
import time
import pandas as pd

device = torch.device('cuda:0')


class Runner:
    def __init__(self, iter_time, **configs):
        super(Runner, self).__init__()
        self.iter_time = iter_time
        self.logger = configs['logger']
        self.configs = configs
        tide_model = TiDE(**configs)
        self.device = device
        self.tide_model = tide_model.to(device) if torch.cuda.is_available() else tide_model

        data_configs = configs['data']
        model_configs = configs['model']
        self.model_parameters = f"train_tide_{data_configs['dataset_name']}_task{data_configs['task_type']}_" \
                      f"in{model_configs['in_lens']}_out{model_configs['out_lens']}_" \
                      f"hid{model_configs['num_hidden']}_ne{model_configs['num_layers_enc']}_" \
                      f"nd{model_configs['num_layers_dec']}_emb{model_configs['embed_type']}_iter{iter_time}"
        self.logger.info(self.model_parameters)

    def _get_data(self, mode):
        data_set, data_loader = data_provider(mode, **self.configs)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = torch.optim.Adam(self.tide_model.parameters(), lr=self.configs['train']['lr'])
        return optimizer

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def valid(self, data_loader, criterion):
        model_configs = self.configs['model']
        total_loss = []
        self.tide_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_timestamp, batch_y_timestamp) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = batch_x_timestamp.float().to(self.device)
                batch_y_timestamp = batch_y_timestamp[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = torch.cat([batch_x_timestamp, batch_y_timestamp], dim=1)

                y_pred = self.tide_model(batch_x, batch_x_timestamp)
                f_dim = -1 if self.configs['data']['task_type'] == 'M2S' else 0
                y_pred = y_pred[:, -self.configs['model']['out_lens']:, f_dim:]
                y_label = batch_y[:, -self.configs['model']['out_lens']:, f_dim:].to(self.device)

                y_pred = y_pred.detach().cpu()
                y_label = y_label.detach().cpu()

                loss = criterion(y_pred, y_label)

                total_loss.append(loss)

            total_loss = np.average(total_loss)
            self.tide_model.train()
            return total_loss

    def train(self):
        _, train_loader = self._get_data(mode='train')
        valid_data, valid_loader = self._get_data(mode='val')
        test_data, test_loader = self._get_data(mode='test')

        data_configs = self.configs['data']
        model_configs = self.configs['model']

        path = os.path.join(self.configs['train']['checkpoints'], self.model_parameters)
        if not os.path.exists(path):
            os.makedirs(path)

        step_start_time = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.configs['train']['patience'], verbose=True, delta=0)

        optimizer = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.configs['train']['epochs'])
        criterion = self._select_criterion()


        for epoch in range(self.configs['train']['epochs']):
            step = 0
            train_loss_list = []

            self.tide_model.train()
            epoch_start_time = time.time()
            for i, (batch_x, batch_y, batch_x_timestamp, batch_y_timestamp) in enumerate(train_loader):
                optimizer.zero_grad()
                step += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = batch_x_timestamp.float().to(self.device)
                batch_y_timestamp = batch_y_timestamp[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = torch.cat([batch_x_timestamp, batch_y_timestamp], dim=1)

                y_pred = self.tide_model(batch_x, batch_x_timestamp)

                f_dim = -1 if data_configs['task_type'] == 'M2S' else 0
                y_pred = y_pred[:, -model_configs['out_lens']:, f_dim:]
                y_label = batch_y[:, -model_configs['out_lens']:, f_dim:].to(self.device)
                loss = criterion(y_pred, y_label)
                train_loss_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titer: {i+1}, epoch: {epoch + 1} | loss:{loss.item():.7f}")
                    speed = (time.time() - step_start_time) / step
                    left_time = speed * ((self.configs['train']['epochs'] - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}")
                    step = 0
                    step_start_time = time.time()

                loss.backward()
                optimizer.step()

            self.logger.info(f"Epoch: {epoch + 1}, cost time: {time.time() - epoch_start_time}")
            train_loss = np.average(train_loss_list)
            valid_loss = self.valid(data_loader=valid_loader, criterion=criterion)
            test_loss = self.valid(data_loader=test_loader, criterion=criterion)

            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {valid_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(valid_loss, self.tide_model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step()

        best_model_path = path + '/' + 'best_checkpoint.pth'
        self.tide_model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, load_models=False):
        _, test_loader = self._get_data(mode='test')
        data_configs = self.configs['data']
        model_configs = self.configs['model']
        if load_models:
            self.tide_model.load_state_dict(state_dict=torch.load(os.path.join('./checkpoints/tide/' + self.model_parameters, 'best_checkpoint.pth')))

        pred_list = []
        label_list = []

        folder_path = './test_results/' + self.model_parameters + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.tide_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_timestamp, batch_y_timestamp) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = batch_x_timestamp.float().to(self.device)
                batch_y_timestamp = batch_y_timestamp[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = torch.cat([batch_x_timestamp, batch_y_timestamp], dim=1)

                y_pred = self.tide_model(batch_x, batch_x_timestamp)

                f_dim = -1 if data_configs['task_type'] == 'M2S' else 0
                y_pred = y_pred[:, -model_configs['out_lens']:, f_dim:]
                y_label = batch_y[:, -model_configs['out_lens']:, f_dim:].to(self.device)
                y_pred = y_pred.detach().cpu().numpy()
                y_label = y_label.detach().cpu().numpy()

                pred_list.append(y_pred)
                label_list.append(y_label)

                # visual
                if i % 20 == 0:
                    input_x = batch_x.detach().cpu().numpy()
                    ground_truth_data = np.concatenate((input_x[0, :, -1], y_label[0, :, -1]), axis=0)
                    predict_data = np.concatenate((input_x[0, :, -1], y_pred[0, :, -1]), axis=0)
                    visual(ground_truth_data, predict_data, os.path.join(folder_path, str(i) + '.pdf'))

        if self.configs['train']['test_flop']:
            test_params_flop(self.tide_model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        pred_list = np.concatenate(pred_list, axis=0)
        label_list = np.concatenate(label_list, axis=0)
        pred_list = pred_list.reshape(-1, pred_list.shape[-2], pred_list.shape[-1])
        label_list = label_list.reshape(-1, label_list.shape[-2], label_list.shape[-1])
        print('test shape:', pred_list.shape, label_list.shape)

        # Result save
        folder_path = './results/' + self.model_parameters + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # shape [num_samples, out_lens, num_nodes||num_features]
        mae, mse, rmse, mape, mspe, rse, corr = metric(pred_list, label_list)
        self.logger.info(f'mse:{mse}, mae:{mae}')
        f = open('result.txt', 'a')
        f.write(self.model_parameters + "  \n")
        f.write(f'mse:{mse}, mae:{mae}, rse:{rse}, corr:{corr}')
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', pred_list)
        # np.save(folder_path + 'true.npy', label_list)


        return

    def predict(self, load_models=False):
        pred_data, pred_loader = self._get_data(mode='pred')
        data_configs = self.configs['data']
        model_configs = self.configs['model']
        if load_models:
            path = os.path.join(self.configs['train']['checkpoints'], self.model_parameters)
            best_model_filename = path + '/' + 'best_checkpoint.pth'
            self.tide_model.load_state_dict(torch.load(best_model_filename))


        pred_list = []

        self.tide_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_timestamp, batch_y_timestamp) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = batch_x_timestamp.float().to(self.device)
                batch_y_timestamp = batch_y_timestamp[:, -model_configs['out_lens']:, :].float().to(self.device)
                batch_x_timestamp = torch.cat([batch_x_timestamp, batch_y_timestamp], dim=1)

                y_pred = self.tide_model(batch_x, batch_x_timestamp)

                y_pred = y_pred.detach().cpu().numpy()
                pred_list.append(y_pred)

        pred_list = np.array(pred_list)
        pred_list = np.concatenate(pred_list, axis=0)

        if pred_data.scale:
            pred_list = pred_data.inverse_transform(pred_list)


        # Result save
        folder_path = '../results/' + self.model_parameters + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', pred_list)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), pred_list, axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return





