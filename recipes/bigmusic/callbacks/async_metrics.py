import logging
import collections
from typing import Any
import multiprocessing.pool as pool
from pytorch_lightning import Callback, Trainer, LightningModule

logger = logging.getLogger(__name__)


class AsyncMetricsCallback(Callback):
    def __init__(self, additional_callbacks=None, callbacks=None) -> None:
        super().__init__()
        self.additional_callbacks = [] if None else additional_callbacks
        self.callbacks = [] if None else callbacks

    @staticmethod
    def async_run_callbacks(callbacks, aggregate_outputs, func, args):
        thread_pool = pool.ThreadPool(processes=len(callbacks))
        async_results = []
        for callback in callbacks:
            async_result = thread_pool.apply_async(func, args=(callback,) + args)
            async_results.append(async_result)
        thread_pool.close()
        for callback, async_result in zip(callbacks, async_results):
            callback_name = callback.__class__.__name__
            callback_output = async_result.get()
            aggregate_outputs[callback_name] = callback_output
        thread_pool.join()

        return aggregate_outputs


    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        aggregate_outputs = collections.OrderedDict()
        args = (trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, )
        aggregate_outputs = self.async_run_callbacks(
            self.additional_callbacks, 
            aggregate_outputs, 
            self._on_predict_batch_end, 
            args,
        )

        aggregate_outputs = self.async_run_callbacks(
            self.callbacks, 
            aggregate_outputs, 
            self._on_predict_batch_end, 
            args,
        )

        return aggregate_outputs
    
    
    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        aggregate_outputs = collections.OrderedDict()
        args = (trainer, pl_module, )
        aggregate_outputs = self.async_run_callbacks(
            self.additional_callbacks, 
            aggregate_outputs, 
            self._on_predict_end, 
            args,
        )

        aggregate_outputs = self.async_run_callbacks(
            self.callbacks, 
            aggregate_outputs, 
            self._on_predict_end, 
            args,
        )

        return aggregate_outputs

    def _on_predict_batch_end(self, callback, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        callback_name = callback.__class__.__name__
        logger.info(f"Callback {callback_name} start 'on_predict_batch_end' ...")
        try:
            out = callback.on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        except NotImplementedError as e:
            logger.warning(f"Callback {callback_name} does not implement 'on_predict_batch_end'")
            out = None
        except Exception as e:
            logger.error(f"Callback {callback_name} raised an exception 'on_predict_batch_end': {e}")
            raise RuntimeError(f"Callback {callback_name} raised an exception 'on_predict_batch_end': {e}")
        logger.info(f"Callback {callback_name} finish 'on_predict_batch_end' ...")
        return out
    
    def _on_predict_end(self, callback:Callback, trainer: Trainer, pl_module: LightningModule):
        callback_name = callback.__class__.__name__
        logger.info(f"Callback {callback_name} start 'on_predict_end' ...")
        try:
            out = callback.on_predict_end(trainer, pl_module)
        except NotImplementedError as e:
            logger.warning(f"Callback {callback_name} does not implement 'on_predict_end'")
            out = None
        except Exception as e:
            logger.error(f"Callback {callback_name} raised an exception 'on_predict_end': {e}")
            raise RuntimeError(f"Callback {callback_name} raised an exception 'on_predict_end': {e}")
        logger.info(f"Callback {callback_name} finish 'on_predict_end' ...")
        return out
