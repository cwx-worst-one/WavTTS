from tests.model_test.metrics.metric_time import metric_time, report_time
from tests.model_test.metrics.metric_gpu_memory import metric_gpu_memory, report_gpu_memory


MetricDict = {'time': metric_time,
              'gpu_memory': metric_gpu_memory}

ReportDict = {'time': report_time,
              'gpu_memory': report_gpu_memory}
