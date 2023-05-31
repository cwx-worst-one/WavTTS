import numpy as np
import matplotlib.pyplot as plt
import os


def plot_bars(args, output_dir, datas, batch_size, max_seqlen, providers,
              xlabel="Batch size", ylabel="", tick_step=1, group_gap=0.2, bar_gap=0.05):
    title = args.model_params.model + ":" + \
        str("{:.3f}".format(args.model_size / 1000000000)) + \
        "B, action:" + args.benchmark_params.action + \
        ", input length:" + str(max_seqlen) + ", " + args.device_type

    output_dir = os.path.join(output_dir, "png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = np.arange(len(batch_size)) * tick_step
    group_num = len(datas)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num
    bar_width = bar_span - bar_gap

    for index, y in enumerate(datas):
        x_ = x + index * bar_span
        plt.bar(x_, y, bar_width, label=providers[index])
        for a, b, i in zip(x_, y, range(len(x_))):
            plt.text(a, b + 0.01, "%.2f" % y[i], ha='center', fontsize=8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.title(title)

    ticks = x + (group_width - bar_span) / 2
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(ticks, batch_size)
    plt.savefig(os.path.join(output_dir, 'seqlen_' +
                str(max_seqlen) + ".png"), bbox_inches='tight')
    plt.close()
