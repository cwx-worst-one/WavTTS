## zh-cn
位于model目录的所有文件包含了算子以及底层模块，算子可以是基于torch的op组装，也可以是自定义的实现。算子均位于layers子文件夹中。

底层模块是基于算子组装而成的上层组件，可以直接被solution使用，要求继承自torch.nn.Module基础类。