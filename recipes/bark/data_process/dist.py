import os
import subprocess
import argparse


def main(args):
    processes = []
    for rank in range(args.world_size):
        my_env = os.environ.copy()
        if rank == 0:
            stdout = None
        else:
            stdout = open(os.devnull, 'w')
        command = [
            'python3',
            args.py_path,
            '--rank={}'.format(rank),
            '--world_size={}'.format(args.world_size),
        ]
        p = subprocess.Popen(command, stdout=stdout, env=my_env)
        processes.append(p)
        print(command)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--py_path', type=str, required=True, help='py_path')
    parser.add_argument('--world_size', type=int, default=1, help='world_size')
    args = parser.parse_args()

    main(args)
