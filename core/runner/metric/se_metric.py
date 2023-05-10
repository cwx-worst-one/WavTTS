'''Se Metric'''
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from core.utils import get_rank, logging
from .base_metric import BaseMetric
from .meter import WeightedMeter, BaseMeter

# pylint: disable=abstract-class-instantiated


class SeMetric(BaseMetric):
    '''Se Metric for speech enhancement

    ::

        Display training loss of se training, if mix loss used,
        display every item(mse ce sisnr). Both current and avg loss are used here.
        Calculation Formula
        average(backward_loss) = sum(backward_loss * target_size) / sum(target_size)
        average(avg_loss) = sum(avg_loss * target_size) / sum(target_size)
        average(cur_mse_loss) = sum(cur_mse_loss * target_size) / sum(target_size)
        average(cur_relmse_loss) = sum(cur_relmse_loss * target_size) / sum(target_size)
        average(cur_sisnr_loss) = sum(cur_sisnr_loss * target_size) / sum(target_size)
        average(cur_ce_loss) = sum(cur_ce_loss * target_size) / sum(target_size)
    '''

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        backward_loss=WeightedMeter,
        avg_loss=WeightedMeter,
        cur_mse_loss=WeightedMeter,
        cur_relmse_loss=WeightedMeter,
        cur_sisnr_loss=WeightedMeter,
        cur_sisnr_imp=WeightedMeter,
        cur_ce_loss=WeightedMeter,
        gnorm=BaseMeter,
    )

    def update(self, data_dict):
        '''update'''
        self._output_buffer.clear()

        assert isinstance(data_dict, dict)
        rank = get_rank()
        skipped_key = []
        for key in data_dict.keys():
            val = data_dict[key]
            count = 1
            if math.isfinite(count) and math.isfinite(val):
                self._add(key, val, count)
            else:
                skipped_key.append(key)
        if len(skipped_key) > 0:
            logging.all_rank_warning(
                'rank %d: sdt_metric: value is not finite for keys: %r', rank, skipped_key
            )


class SslTrainMetric(SeMetric):
    """SSL Metric for training

    based on SeMetric,but _name2type only include time, data_time, backward_loss, gnorm
    """

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        backward_loss=WeightedMeter,
        gnorm=BaseMeter,
    )


class DenoiseTrainMetric(SeMetric):
    """Denoise Metric for training

    based on SeMetric,but _name2type only include time, data_time, backward_loss, gnorm
    """

    _name2type = dict(
        time=BaseMeter,
        data_time=BaseMeter,
        backward_loss=WeightedMeter,
        gnorm=BaseMeter,
    )


class BfInferMetricSpatialResponseAndSisnr(BaseMetric):
    """Bf Metric for inference

    only used to plot beam pattern
    """

    def update(self, data_dict):
        """
        update
        data_dict: {
           net_direction: {
               sou_direction: {
                      energy_response_ovlp: float,
                      energy_response_nonovlp: float,
                      count_ovlp: float,
                      count_nonovlp: float,
                      sisnr_imp: float,
                      sisnr_count: int
                   }
           }
        }
        """
        # pylint:disable=too-many-branches
        self._output_buffer.clear()
        rank = get_rank()
        skipped_key = []
        for net_direction in data_dict.keys():
            sou_dict = data_dict[net_direction]
            for sou_direction in sou_dict.keys():
                res = sou_dict[sou_direction]
                name = f"{net_direction}_{sou_direction}"
                # ovlp
                if res.get('count_ovlp', None) is not None:
                    count = res['count_ovlp']
                    if count != 0:
                        name1 = "ovlp_" + name
                        if name1 not in self._meters:
                            self.add_meter(name1, WeightedMeter)
                        val = res['energy_response_ovlp'] / count if count != 0 else 0
                        if math.isfinite(count) and math.isfinite(val):
                            self._add(name1, val, count)
                        else:
                            skipped_key.append(name1)
                # nonovlp
                if res.get('count_nonovlp', None) is not None:
                    count = res['count_nonovlp']
                    if count != 0:
                        name2 = "nonovlp_" + name
                        if name2 not in self._meters:
                            self.add_meter(name2, WeightedMeter)
                        val = res['energy_response_nonovlp'] / count if count != 0 else 0
                        if math.isfinite(count) and math.isfinite(val):
                            self._add(name2, val, count)
                        else:
                            skipped_key.append(name2)
                # sisnr
                count = res['sisnr_count']
                val = res['sisnr_imp']
                if val is not None:
                    val = val / count
                    name3 = "sisnr_" + name
                    if name3 not in self._meters:
                        self.add_meter(name3, WeightedMeter)
                    if math.isfinite(count) and math.isfinite(val):
                        self._add(name3, val, count)
                    else:
                        skipped_key.append(name3)
        if len(skipped_key) > 0:
            logging.all_rank_warning(
                'rank %d: sdt_metric: value is not finite for keys: %r', rank, skipped_key
            )

    def process(self, save_dir):
        """generate beam pattern plots and xlsx"""
        out_files = []
        message = self.get()
        message_keys_ovlp = list(filter(lambda x: x.startswith("ovlp"), message.keys()))
        message_keys_nonovlp = list(filter(lambda x: x.startswith("nonovlp"), message.keys()))
        message_keys_sisnr = list(filter(lambda x: x.startswith("sisnr"), message.keys()))
        for info_ori in [message_keys_ovlp, message_keys_nonovlp, message_keys_sisnr]:
            if len(info_ori) == 0:
                continue
            info = list(map(lambda x: x.split('_'), info_ori))
            wav_type = info[0][0]
            net_direction = {x[1] for x in info}
            src_direction = {x[2] for x in info}
            net_direction = [round(float(x), 2) for x in list(net_direction)]
            net_direction.sort()
            src_direction = [round(float(x), 2) for x in list(src_direction)]
            src_direction.sort()
            res = np.ones((len(net_direction), len(src_direction)))
            for key in info_ori:
                value = message[key]
                net_d, src_d = key.split('_')[1:]
                net_idx = net_direction.index(float(net_d))
                src_idx = src_direction.index(float(src_d))
                res[net_idx, src_idx] = value
            if wav_type == "ovlp" or wav_type == "nonovlp":  # pylint: disable=consider-using-in
                # energy response
                energy_res = 10 * np.log10(res + 1e-6)
                energy_res = np.clip(energy_res, a_min=-25.5, a_max=1.5)
                # energy res plot
                fig, ax = plt.subplots()
                cs = ax.imshow(energy_res, aspect="auto")
                fig.colorbar(cs)
                ax.set_yticks(range(0, len(net_direction)))
                ax.set_yticklabels([f"{x:.1f}" for x in net_direction])
                ax.set_xticks(range(0, len(src_direction)))
                ax.set_xticklabels([f"{x:.1f}" for x in src_direction])
                ax.set_xlabel("src direction")
                ax.set_ylabel("net direction")
                plt.savefig(os.path.join(save_dir, f'energy_response_{wav_type}.png'))
                plt.close()
                out_files.append(os.path.join(save_dir, f'energy_response_{wav_type}.png'))
                # save xlsx
                try:
                    df = pd.DataFrame(
                        energy_res.tolist(), index=net_direction, columns=src_direction
                    )
                    writer = pd.ExcelWriter(
                        os.path.join(save_dir, f'energy_response_{wav_type}.xlsx')
                    )
                    df.to_excel(
                        writer,
                        sheet_name='energy_response',
                        index_label='net_direction',
                        float_format="%.2f",
                    )
                    writer.save()
                    out_files.append(os.path.join(save_dir, f'energy_response_{wav_type}.xlsx'))
                except ModuleNotFoundError as e:
                    print(e)
            else:
                # sisnr
                try:
                    df = pd.DataFrame(res.tolist(), index=net_direction, columns=src_direction)
                    writer = pd.ExcelWriter(os.path.join(save_dir, 'sisnr.xlsx'))
                    df.to_excel(
                        writer,
                        sheet_name='sisnr improvement',
                        index_label='net_direction',
                        float_format="%.2f",
                    )
                    writer.save()
                    out_files.append(os.path.join(save_dir, 'sisnr.xlsx'))
                except ModuleNotFoundError as e:
                    print(e)
        return out_files


class SslInferMetricSaveAcc(BaseMetric):
    """Ssl Metric for inference

    used to count ssl accuracy"""

    def update(self, data_dict):
        # data_dict: {
        #    data_direction: {
        #        diff_bins: [int, int, ...],
        #        total_cnt: int,
        #        accurate_cnt: int
        #    },
        #    bins: list[float]
        # }
        '''update'''
        self._output_buffer.clear()
        rank = get_rank()
        skipped_key = []
        bins = data_dict['bins']
        data_dict_keys = list(data_dict.keys())
        data_dict_keys.remove('bins')
        for data_direction in data_dict_keys:
            res = data_dict[data_direction]
            count = res['total_cnt']
            # accuracy
            name = "accuracy_" + data_direction
            val = res['accurate_cnt'] / float(count)
            if name not in self._meters:
                self.add_meter(name, WeightedMeter)
            if math.isfinite(count) and math.isfinite(val):
                self._add(name, val, count)
            else:
                skipped_key.append(name)
            # diff_histogram
            for bin_idx in range(len(bins) - 1):
                name = (
                    'hist_' + data_direction + "_" + f"{bins[bin_idx]}_{bins[bin_idx+1]}_{bin_idx}"
                )
                val = res['diff_bins'][bin_idx] / float(count)
                if name not in self._meters:
                    self.add_meter(name, WeightedMeter)
                if math.isfinite(count) and math.isfinite(val):
                    self._add(name, val, count)
                else:
                    skipped_key.append(name)
            # hist_bin_num
        name = 'bin_num'
        val = float(len(bins) - 1)
        if name not in self._meters:
            self.add_meter(name, WeightedMeter)
        self._add(name, val, 1)
        if len(skipped_key) > 0:
            logging.all_rank_warning(
                'rank %d: sdt_metric: value is not finite for keys: %r', rank, skipped_key
            )

    def process(self, save_dir):
        """generate accuracy"""
        out_files = []
        message = self.get()
        bin_num = int(round(message['bin_num']))
        # pylint:disable=consider-using-with
        txt_file = open(os.path.join(save_dir, "accuracy.txt"), 'wt', encoding='utf-8')
        accuracy_list = []
        hist_dict = dict()
        bin_name = dict()
        message_keys = list(message.keys())
        message_keys.remove('bin_num')
        for info_ori in message_keys:
            info = info_ori.split('_')
            key_type = info[0]
            direction = float(info[1])
            if key_type == "accuracy":
                accuracy_list.append((float(direction), message[info_ori]))
            elif key_type == "hist":
                bin_idx = str(int(info[-1]))
                if bin_idx not in bin_name:
                    bin_low = float(info[2])
                    bin_high = float(info[3])
                    bin_name[bin_idx] = f"[{bin_low:.2f}, {bin_high:.2f})"
                if info[1] not in hist_dict:
                    hist_dict[f"{direction:.2f}"] = np.zeros(bin_num)
                hist_dict[f"{direction:.2f}"][int(bin_idx)] = message[info_ori]
        accuracy_list = sorted(accuracy_list, key=lambda x: x[0])
        acc_all = 0
        for direction, acc_res in accuracy_list:
            info1 = f"histogram for {direction:.2f}: "
            hist_res = hist_dict[f"{direction:.2f}"] * 100
            for bin_idx in range(len(hist_res)):  # pylint: disable=consider-using-enumerate
                info1 = info1 + bin_name[str(int(bin_idx))] + f"-{hist_res[bin_idx]:.2f}%; "
            acc_res = acc_res * 100
            info2 = f"accuracy for {direction:.2f}: {acc_res:.2f}%"
            print(info1, file=txt_file)
            print(info2, file=txt_file)
            print("\n", file=txt_file)
            acc_all += acc_res
        acc_all = acc_all / len(accuracy_list)
        info = f"accuracy for whole dataset: {acc_all:.2f}%"
        print(info, file=txt_file)
        txt_file.close()
        out_files.append(os.path.join(save_dir, "accuracy.txt"))
        return out_files


class DenoiseInferMetric(BaseMetric):
    """Denoise Metric for inference

    used to count Denoise accuracy"""

    _name2type = dict(pesq=BaseMeter, sisnr=BaseMeter)

    def update(self, data_dict):
        '''cal pesq update'''
        self._output_buffer.clear()
        pesq = data_dict.get("pesq", 0)
        self._add('pesq', pesq, 1)
        sisnr = data_dict.get("sisnr", 0)
        self._add('sisnr', sisnr, 1)


class AECInferMetric(BaseMetric):
    """AEC Metric for inference

    used to count AEC accuracy"""

    _name2type = dict(pesq=BaseMeter, reduce_db=BaseMeter)

    def update(self, data_dict):
        '''cal pesq update'''
        self._output_buffer.clear()
        pesq = data_dict.get("pesq", 0)
        self._add('pesq', pesq, 1)
        sisnr = data_dict.get("reduce_db", 0)
        self._add('reduce_db', sisnr, 1)
