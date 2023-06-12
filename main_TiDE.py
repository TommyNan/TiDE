import argparse
import numpy as np
import random
import torch
import yaml
from lib.util import get_logger_simple
from model.trainer import Runner


def main(args):
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
        config['logger'] = logger

        if config['train']['is_training']:
            total_iter_times = config['train']['itr']
            for iter_time in range(total_iter_times):
                runner = Runner(iter_time, **config)

                logger.info(f'>>>>>>>start training - iter: {iter_time}/{total_iter_times}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.train()

                logger.info(f'>>>>>>>testing - iter: {iter_time}/{total_iter_times}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.test(load_models=False)

                if config['train']['do_predict']:
                    logger.info(f'>>>>>>>predicting - iter: {iter_time}/{total_iter_times}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    runner.predict(load_models=True)

                torch.cuda.empty_cache()

        else:
            iter_time = 0
            runner = Runner(iter_time, **config)
            if config['train']['do_predict']:
                logger.info(f'>>>>>>>predicting>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.predict(load_models=True)

            else:
                logger.info(f'>>>>>>>testing - iter: {iter_time}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                runner.test(load_models=True)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # generate parser and args
    parser = argparse.ArgumentParser('Baseline-TiDE')
    parser.add_argument('--dataset', default='ETTh2', choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ELEC', 'EXCHANGE',
                                                               'TRAFFIC', 'WEATHER', 'ILI'])
    args = parser.parse_args()
    args.config_filename = f'./config/TiDE/TiDE_{args.dataset}.yaml'

    # generate loggers in the specific dir
    logger = get_logger_simple('log', f'TiDE_{args.dataset}')
    print(f'Baseline:TiDE\t-Dataset:{args.dataset}')
    main(args)
