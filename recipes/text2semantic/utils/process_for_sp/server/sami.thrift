include "base.thrift"

namespace go lab.sami
namespace py lab.sami

struct InvokeRequest {
    1: required string access_key,      // user access key, 用户的凭证
    2: required string method,          // method namespace, 方法命名空间
    3: optional string payload,         // json payload, 文本数据
    4: optional binary data,            // binary data, 二进制数据
    5: required string version = "v4",  // version, 协议版本，用户无需指定

    6: optional string task_id = "",    // task id, 客户端可设置task id，需要使用uuid保证唯一性
    7: optional bool is_offline = false,// is offline, 指定请求任务是否为离线任务，若为true则会进行离线调度

    255: required base.Base Base,
}

struct InvokeResponse {
    1: required string task_id,         // task id, 本次调用的全局唯一识别码，提交工单必须附加
    2: optional string payload,         // returned payload, 返回的文本数据
    3: optional binary data,            // returned binary data, 返回的二进制数据
    4: optional string state,           // state, 离线任务执行状态

    255: required base.BaseResp BaseResp,
}

service SamiService {
    InvokeResponse Invoke(1: InvokeRequest req)
}

service SAMI {
    InvokeResponse Invoke(1: InvokeRequest req)
}
