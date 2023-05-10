## Dataset
`HDFSDataset` and `ValidHDFSDataset` are provided to read dataset from HDFS.
Both of them support bucket shechdule, item transform and batch transform.
### HDFSDataset
Adavantages:
 - provide high performance by assigning works to multi child process.

 DisAdavantages:
 - log will be mixed and hardly understood when error happens. When Errors happen on a process,
 communications between process will fail too, and many 'Broken Pipe' will be reported.
 - Child process may stays when you try to kill this trial by 'Ctrl + C'.
 Main process will be terminated directly, no time to notify child process.
### DebugHDFSDataset
Advantage:
 - It's simple, and easily to do debug.

Disadvantage:
 - The latency to get a batch data may be bigger.

### Batch Size
Batch size of short sentence(less number of frame) could be bigger than current config when batch means tocken set.

For example, batch size for RNNT Model on tel & video dataset is 15360.

But the batch size for sentence that 100 < #frame < 200 could be 50000.

So we provide a config option to let user to set different max batch size for differenct bucket(different #frame).

Config option `data.max_batch_scale` is provided. Its default value is 0, which means no scale for batch size.

max batch size for each bucket is:`(max_bucket_value - current_bucket_value) * max_batch_scale + max_batch_size`.

`data.max_batch_scale` should be carefully set, suggested value is 10.

## Preprocess
`draw_batch_fn` and `parse_fn` are splited into basic functions, So they can be reused.

They all organized by callable class.

`draw_batch_fn` and `parse_fn` can be built from config file, and easily to use.
