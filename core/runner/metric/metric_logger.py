''' logger module '''
import os
import datetime
from typing import OrderedDict

from core.utils import logging, hdfs_mkdir, hdfs_put
from core.runner.utils import get_max_memory
from core.runner.metric.base_metric import BaseMetric
from core.runner.tensorboard import SummaryWriter

try:
    import falconclaw
except ImportError:
    falconclaw = None


class BaseMetricLogger:
    '''base logger'''

    def __init__(self, interval=10, rank=-1):
        '''init.'''
        self.interval = interval
        self.rank = rank

    def every_interval_iters(self, iters):
        '''evary interval iters'''
        # NOTE: iters should start at 0
        return (iters + 1) % self.interval == 0 if iters >= 0 and self.interval > 0 else False

    @staticmethod
    def preprocess(runner_msg, message):
        '''preprocess'''
        log_dict = runner_msg.copy()
        if isinstance(message, BaseMetric):
            message = message.get()
        if log_dict['name'] is not None:
            log_dict['mode'] = log_dict['name']
        del log_dict['name']
        if runner_msg['mode'] == 'train':
            log_dict['eta'] = message['time'] * runner_msg['remain_steps']
            del log_dict['remain_steps']
            log_dict['time'] = message['time']
            log_dict['data_time'] = message['data_time']
            log_dict['memory'] = get_max_memory()
        for key, value in message.items():
            if key not in ['time', 'data_time']:
                log_dict[key] = value
        return log_dict

    def log(self, runner_msg, message):
        '''log'''
        # NOTE: runner_msg is a dict which have mode, epoch, and
        # iter values. epoch, iter should start at 1

    def report(self, message):
        '''report'''

    def flush(self):
        '''Flush to tensorboard.'''

    def close(self):
        '''Close writer of tensorboard.'''


class MetricLogger(BaseMetricLogger):
    '''stdout logger'''

    SKIPS = ['mode', 'epoch', 'iter']

    @staticmethod
    def preprocess(runner_msg, message):
        '''preprocess'''
        message = BaseMetricLogger.preprocess(runner_msg, message)
        log_dict = OrderedDict()
        for key, value in message.items():
            if key == "lr":
                if isinstance(value, list):
                    log_dict[key] = str([f'{v:.2e}' for v in value])
                else:
                    log_dict[key] = f'{value:.2e}'
            elif key == "eta":
                log_dict[key] = str(datetime.timedelta(seconds=int(value)))
            elif key in ['time', 'data_time']:
                log_dict[key] = f'{value:.3f}'
            elif key == 'memory':
                log_dict[key] = f'{value:.1f}M'
            else:
                log_dict[key] = value
        return log_dict

    def log(self, runner_msg, message):
        if runner_msg['mode'] == 'train' and not self.every_interval_iters(runner_msg['iter'] - 1):
            return
        message = self.preprocess(runner_msg, message)
        log_str = f'Epoch({message["mode"]}) [{message["epoch"]}]'
        log_str += f'[{message["iter"]}] ' if 'iter' in message else ' '
        log_items = []
        for name, val in message.items():
            if name in self.SKIPS:
                # since SKIPS has been specially dealt, repeated processing is avoided here
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        logging.info(log_str)


class TensorBoardLogger(BaseMetricLogger):
    '''Tensorboard logger

    Add all var in runner_msg to tensorboard.
    '''

    SKIPS = ['mode', 'epoch', 'iter']

    def __init__(self, interval=10, rank=-1, flush_steps=260):
        super().__init__(interval, rank)
        self.flush_steps = (flush_steps + interval - 1) // interval * interval
        logging.info(f"Tensorboard, add_scalar freq: {interval}, flush freq: {self.flush_steps}.")
        # Init tensorboard writer.
        # If current env is arnold env, the tensorboard is flush to local dir
        # first, then push to hdfs later. arnold env include arnold trial
        # run mode and debug mode.
        # For other env such as workspace, the tensorboard is flush to local
        # dir only. Different run will be seperated by timestamp.
        arnold_output = os.environ.get('ARNOLD_OUTPUT', 'not_arnold_env')
        self.remote_enabled = (arnold_output != 'not_arnold_env') and (self.rank == 0)
        self.local_enabled = self.rank == 0

        if self.local_enabled:
            timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            self.local_tb_dir = f'./tensorboard/{timestamp}'
            os.makedirs(self.local_tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.local_tb_dir)
            logging.info(f"Tensorboard local enabled, will be saved to {self.local_tb_dir}.")

        if self.remote_enabled:
            self.remote_tb_dir = '{}/tb_{}'.format(
                os.environ["ARNOLD_OUTPUT"], os.environ["ARNOLD_TRIAL_ID"]
            )
            hdfs_mkdir(self.remote_tb_dir)
            logging.info(f"Tensorboard remote enabled, will be saved to {self.remote_tb_dir}.")

    @staticmethod
    def preprocess(runner_msg, message):
        '''preprocess'''
        raw_message = BaseMetricLogger.preprocess(runner_msg, message)
        message = OrderedDict()
        for key, value in raw_message.items():
            if key == "lr":
                if isinstance(value, list) and isinstance(value[0], (float, int)):
                    for i, v in enumerate(value):
                        message["{}{}".format(key, i)] = float(v)
                elif isinstance(value, (float, int)):
                    message[key] = float(value)
            elif isinstance(value, (float, int)):
                message[key] = value
        return message

    def log(self, runner_msg, message):
        '''Save var to tensorboard.'''
        if not self.remote_enabled and not self.local_enabled:
            return
        if runner_msg['mode'] == 'train' and not self.every_interval_iters(runner_msg['iter'] - 1):
            return
        prefix = 'train' if runner_msg['mode'] == 'train' else 'val'
        message = self.preprocess(runner_msg, message)
        for k, v in message.items():
            if k in self.SKIPS:
                continue
            self.writer.add_scalar(f'{prefix}/{k}', v, runner_msg['iter'])
        if runner_msg['iter'] % self.flush_steps == 0:
            self.flush()
            logging.info(f"Tensorboard, flush success at iter: {runner_msg['iter']}")

    def flush(self):
        if not self.remote_enabled and not self.local_enabled:
            return
        # Save to local_tb_dir.
        self.writer.flush()
        if self.remote_enabled:
            hdfs_put(self.local_tb_dir + "/*", self.remote_tb_dir, retry=3, timeout=3 * 60)  # 3min

    def close(self):
        if not self.remote_enabled and not self.local_enabled:
            return
        # Close writer.
        self.flush()
        self.writer.close()


class FalconMetricLogger(BaseMetricLogger):
    '''output to falconclaw.log'''

    SKIPS = ['eta']

    def log(self, runner_msg, message):
        mode, iters = runner_msg['mode'], runner_msg['iter']
        if mode == 'train' and not self.every_interval_iters(iters - 1):
            return
        message = BaseMetricLogger.preprocess(runner_msg, message)
        if self.rank != 0:
            return
        try:
            for skip in self.SKIPS:
                if skip in message:
                    del message[skip]
            falconclaw.log(message, step=iters)
        except Exception:
            pass

    def report(self, message):
        '''report'''
        if self.rank != 0:
            return
        try:
            falconclaw.log_report(message)
        except Exception:
            pass
