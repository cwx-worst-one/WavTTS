# FAQs:
# Q:  为什么这里的 base.thrift 与 service_rpc/idl/base.thrift 不一致？
# A:  service_rpc/idl/base.thrift 生成的 C++ 代码是无法编译通过的。
#     TrafficEnv 使用相同的字段类型和字段名，C++ 中有名字冲突；Client 字段在 fbthrift 中是保留字。
#
# Q:  使用这里的 base.thrift 与 service_rpc/idl/base.thrift 是否兼容？
# A:  使用 BinaryProtocol 传输是完全兼容的，而你也不应该使用其他的 Protocol。
#
# Q:  golang 代码可以依赖这里的 base.thrift 吗？
# A:  可以，golang 生成的代码大小写是依据 golang 的语法，而不是 idl 的定义。
#     但是基于 kite 和 data/idl 在路径约定方面的差异，可能你会碰到一些困难。
#
# Q:  python 代码可以依赖这里的 base.thrift 吗？
# A:  可以。如果依赖了 data/idl 生成的 python 代码，后续要迁移到 pie 框架时需要注意字段名字的改变。
#     由于 python 不是编译执行的，这样的错误可能会发生在运行时。
#
# Q:  我的 idl 既要在 C++ 中使用，又要在 python/golang 中使用，怎么办？
# A:  既提交到 data/idl，又提交到 service_rpc/idl。在两个库中，依赖各自的 base.thrift。
#
# Q:  为什么要维护两份 idl，不能统一吗？
# A:  如果你尝试过然后失败了，请把下面的计数器 +1。
#     4

namespace cpp base
namespace py base
namespace go base
namespace java base

struct TrafficEnv {
    1: bool Open = false,
    2: string Env = "",
}

struct Base {
    1: string LogID = "",
    2: string Caller = "",
    3: string Addr = "",
    4: string client = "",
    5: optional TrafficEnv trafficEnv,
    6: optional map<string, string> extra,
}

struct BaseResp {
    1: string StatusMessage = "",
    2: i32 StatusCode = 0,
    3: optional map<string, string> Extra,
}


