'''
The entrance of weekly test.
'''

from fire import Fire

from report import generate_report
from weekly_test import run_weekly_test


def main(
        work_type: str = 'workflow',
        workflow_type: str = 'both',
        train_task_id: int = 2612535,
        decode_task_id: int = 2612598,
        onnx_task_id: int = 2612531,
        panther_version: str = '1.6.6.0',
        group_ids: str = '52',
        cluster_id: int = 17,
        run_train: int = 1,
        run_decode: int = 1,
        run_onnx: int = 1,
):
    '''
    work_type to do:
    'workflow': start the weekly test
    'report': generate the weekly test report
    'robot': start the robot to send the weekly test condition
    '''
    if work_type == 'workflow':
        if not work_type:
            raise Exception('Please specify workflow_type')
        run_weekly_test(
            train_task_id,
            decode_task_id,
            onnx_task_id,
            panther_version,
            str(group_ids),
            int(cluster_id),
            run_train,
            run_decode,
            run_onnx,
        )
    elif work_type == 'report':
        generate_report(workflow_type)
    else:
        raise ValueError(f"Incorrect work_type: {work_type}")


if __name__ == '__main__':
    Fire(main)
