"""BestrqPretrainRunner"""

import os
import os.path as osp
import numpy as np
import torch
from core.utils import logging
from core.runner.asr.base_asr_runner import BaseAsrRunner
from ..base_runner import RUNNERS
from core.dataset import (
    ValidHDFSDataset,
    build_item_augmentation,
    build_draw_batch_fn,
    build_device_augmentation,
)
from core.utils import (
    dist_barrier,
    get_rank,
    get_local_rank,
    dist_allreduce,
    hdfs_mkdir,
    hdfs_test,
    hdfs_put,
    ReduceOp,
    logging,
)


@RUNNERS.register_module()
class AcousticRepresentationDumpingRunner(BaseAsrRunner):
    """AcousticRepresentationDumpingRunner"""

    def build_dataset(self, dataset_cfg):
        """build data set"""

        self.pre_build_dataset(dataset_cfg)

        logging.info("For inference, we build dataset in validation setting.")

        parse_eval_cfg = dataset_cfg.valid_item_transform
        self.parse_fn_eval = build_item_augmentation(parse_eval_cfg, self.meta_data)

        # draw batch fn
        draw_batch_cfg = dataset_cfg.get("batch_transform", [])
        self.draw_batch_fn = build_draw_batch_fn(draw_batch_cfg, self.meta_data)
        draw_valid_batch_cfg = dataset_cfg.get("valid_batch_transform", draw_batch_cfg)
        self.draw_valid_batch_fn = build_draw_batch_fn(draw_valid_batch_cfg, self.meta_data)

        # device transform
        device_trans_cfg = dataset_cfg.get("device_transform", [])
        self.device_trans = build_device_augmentation(device_trans_cfg, self.meta_data)
        valid_device_trans_cfg = dataset_cfg.get("valid_device_transform", device_trans_cfg)
        self.valid_device_trans = build_device_augmentation(valid_device_trans_cfg, self.meta_data)

        self.get_data_list(dataset_cfg)
        if hasattr(dataset_cfg, "bucket_schedule_val"):
            val_bucket_schedule = dataset_cfg.bucket_schedule_val
        else:
            val_bucket_schedule = dataset_cfg.bucket_schedule

        split_path_list_by_rank = dataset_cfg.get("split_path_list_by_rank", 1)
        self.valid_data_loader = ValidHDFSDataset(
            self.valid_file_list,
            val_bucket_schedule,
            dataset_cfg,
            self.parse_fn_eval,
            self.draw_valid_batch_fn,
            device_transforms=self.valid_device_trans,
            split_path_list_by_rank=split_path_list_by_rank,
            split_each_dataset=False,
        )

    @torch.no_grad()
    def inference_iteration(self):
        """inference for one test dataset"""
        # pylint:disable=too-many-branches
        self.solution.eval()

        self.valid_data_loader.reset()
        batch_data = self.valid_data_loader.next()

        extraction_mode = self.args.inference.get("extraction_mode", "extract_encoder_backbone_out")
        saving_interval = self.args.inference.get("saving_interval", 1)
        log_interval = self.args.log_config.interval

        predicted = []
        iter_i = 0
        while batch_data is not None:
            with torch.no_grad():
                (
                    data,
                    lengths,
                ) = self.solution.extract_data(batch_data, extraction_mode=extraction_mode)
            data = data.detach().cpu().numpy()
            lengths = lengths.detach().cpu().numpy()
            uttid = batch_data["uttid"]
            predicted.append((uttid, data, lengths))
            iter_i += 1
            if iter_i % saving_interval == 0:
                self.save_shard(predicted)
                predicted = []
                logging.info(
                    "Rank %d: Saved %d utterances, %d frames.",
                    get_rank(),
                    self.total_utt,
                    self.total_frames,
                )
            if iter_i % log_interval == 0:
                logging.info("Rank %d: Iterated batch %d", get_rank(), iter_i)

            batch_data = self.valid_data_loader.next()
        if predicted:
            self.save_shard(predicted)

        self.valid_data_loader.terminate()

    def save_shard(self, predicted):
        """save results local"""
        utts = []
        data_list = []
        lengths_list = []
        for utt, batch_data, lengths in predicted:
            utts.extend(utt)
            lengths_list.append(lengths)
            for i in range(lengths.shape[0]):
                one_data = batch_data[i, : lengths[i]]
                data_list.append(one_data)

        data = np.concatenate(data_list, axis=0)
        lengths = np.concatenate(lengths_list, axis=0)
        handler_idx = self.saving_cnt % len(self.saving_handlers)
        (
            fp_utt,
            fp_data,
            fp_lengths,
        ) = self.saving_handlers[handler_idx]

        fp_utt.write("{}\n".format("\n".join(utts)))
        fp_utt.flush()
        fp_data.append(data)
        fp_lengths.append(lengths)

        self.total_frames += np.sum(lengths)
        self.total_utt += lengths.shape[0]
        self.saving_cnt += 1

    def _init_file_handlers_for_saving(self):
        # pylint: disable=too-many-branches
        infer_cfg = self.args.inference
        override_directory = infer_cfg.get("override_directory", False)
        local_saving_dir = infer_cfg.get("local_saving_dir", "./results")
        if self.rank == 0:
            if os.path.exists(local_saving_dir):
                if not override_directory:
                    raise ValueError(
                        "{} already exists! And override_directory==False. ".format(
                            local_saving_dir
                        )
                        + "Thus, we do not execute the inference program."
                    )
                else:
                    logging.info(
                        "{} already exists! We delete the directory to generate new files.".format(
                            local_saving_dir
                        )
                    )
                    os.system("rm -rf " + local_saving_dir)
            os.system("mkdir -p " + local_saving_dir)
            logging.info("Made dir: {}".format(local_saving_dir))
        dist_barrier()

        self.saving_handlers = []
        self.n_shard_per_proc = infer_cfg.get("n_shard_per_proc", 1)
        self.total_utt = 0
        self.total_frames = 0
        self.saving_cnt = 0  # count saving number for shard

        from npy_append_array import NpyAppendArray

        # This pkg is only used once. And this is not in standard
        # dolphin image, so we import it here.
        for shard_i in range(self.n_shard_per_proc):
            self.saving_handlers.append(
                (
                    open(
                        osp.join(local_saving_dir, "rank{}_shard{}_utts.txt").format(
                            get_rank(), shard_i
                        ),
                        "a",  # Here, we use append mode. Because NpyAppendArray is in the append mode.
                    ),
                    NpyAppendArray(
                        osp.join(
                            local_saving_dir,
                            "rank{}_shard{}_data.npy",
                        ).format(get_rank(), shard_i)
                    ),
                    NpyAppendArray(
                        osp.join(
                            local_saving_dir,
                            "rank{}_shard{}_lengths.npy",
                        ).format(get_rank(), shard_i)
                    ),
                )
            )

    def _close_file_handlers(self):
        for f1, f2, f3 in self.saving_handlers:
            f1.close()
            f2.close()
            f3.close()

    @torch.no_grad()
    def inference(self):
        """inference."""
        # pylint: disable=too-many-branches

        self.resume()
        inference_cfg = self.args.inference

        self.get_data_list(self.dataset_cfg)
        self.build_dataset(self.dataset_cfg)

        self._init_file_handlers_for_saving()
        self.inference_iteration()
        dist_barrier()

        logging.info(
            "Rank %d: Saved %d utterances, %d frames.",
            get_rank(),
            self.total_utt,
            self.total_frames,
        )

        total_utt = torch.as_tensor(self.total_utt, dtype=torch.int64, device="cuda")
        dist_allreduce(total_utt, name="total_utt", op=ReduceOp.SUM)
        total_utt = total_utt.cpu().item()
        total_frames = torch.as_tensor(self.total_frames, dtype=torch.int64, device="cuda")
        dist_allreduce(total_frames, name="total_frames", op=ReduceOp.SUM)
        total_frames = total_frames.cpu().item()
        if get_rank() == 0:
            logging.info(
                "In total: Saved %d utterances, %d frames.", self.total_utt, self.total_frames
            )

        dist_barrier()
        self._close_file_handlers()

        remote_saving_dir = inference_cfg.get("remote_saving_dir", "")
        if remote_saving_dir and get_rank() == 0:
            if hdfs_test(remote_saving_dir, "-f") == 0:
                logging.warning(
                    "remote_saving_dir="
                    + remote_saving_dir
                    + " is a file, changed to "
                    + remote_saving_dir
                    + "_dir"
                )
                remote_saving_dir = remote_saving_dir + "_dir"
            if hdfs_test(remote_saving_dir, "-e") != 0:
                hdfs_mkdir(remote_saving_dir)
            local_saving_dir = inference_cfg.get("local_saving_dir", "./results")
            hdfs_put(local_saving_dir, remote_saving_dir, sync=True)
        logging.info("Rank {}: inference finished.".format(self.rank))
