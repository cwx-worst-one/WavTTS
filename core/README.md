### zh
core目录负责管理算法框架中的核心代码，遵循模块化设计原则，细粒度的模块能够增加代码复用，提高代码维护效率。
- dataset 数据集，支持scp、lmdb、hdfs等方式
- utils  工具方法，包括配置、文件读取、配置导出，logger。
- models torch.nn.Module的集合，包含所有的网络组建模块。
- runner 执行器框架入口
- solutions 算法方案集合，比如las/dfsmn/rnnt/kws。 