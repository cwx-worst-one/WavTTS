include "base.thrift"

namespace go lab.sami
namespace py lab.sami

typedef string Priority

// High 有实时性要求
const Priority High = "high"
// Middle 一般的离线任务，允许一定延迟执行
const  Priority Middle = "middle"
// Low 低优先级，例如刷库
const Priority Low = "low"

struct InvokeRequest {
    1: required string access_key,      // user access key, 用户的凭证
    2: required string method,          // method namespace, 方法命名空间
    3: optional string payload,         // json payload, 文本数据
    4: optional binary data,            // binary data, 二进制数据
    5: required string version = "v4",  // version, 协议版本，用户无需指定

    6: optional string task_id = "",    // task id, 客户端可设置task id，需要使用uuid保证唯一性
    7: optional bool is_offline = false,// is offline, 指定请求任务是否为离线任务，若为true则会进行离线调度
    8: optional string token = "",      // token http改造加入，内部调用不鉴权不使用此字段，对外鉴权时会用到

    51: optional Priority priority,      // 当为离线任务时，需要指定任务优先级，如果不指定，默认为middle
    52: optional string batch_task_id,   // 批量任务的task_id
    53: optional i64 batch_index,        // 批量任务的批次, 必须从0开始
    54: optional list<InvokeRequest> batch_req, // 批量请求体
    55: optional bool need_detail,       // 如果是true，会返回未完成的子task_id

    255: required base.Base Base,
}

struct InvokeResponse {
    1: required string task_id,         // task id, 本次调用的全局唯一识别码，提交工单必须附加
    2: optional string payload,         // returned payload, 返回的文本数据
    3: optional binary data,            // returned binary data, 返回的二进制数据
    4: optional string state,           // state, 离线任务执行状态
    10: optional i64 success,            // 批量任务的成功数
    11: optional i64 failed,            // 批量任务的失败数
    12: optional i64 unfinished,        // 批量任务的未完成数
    13: optional list<string> unfinished_task, // 批量任务的未完成具体任务
    14: optional list<string> failed_task, // 批量任务失败的任务
    20: optional i64 total,            // 批量任务的总数

    255: required base.BaseResp BaseResp,
}

const string GetTokenRequestVersionAuthV1 = "auth-v1" // 在线 token
const string GetTokenRequestVersionOfflineAuthV1 = "offline-auth-v1" // 离线 token（包含「在离线 token」）

struct GetTokenRequest {
    1: required string version; // 版本号，类型见上方定义
    2: required string access_key, // 用户的 access_key，可以在 SAIL 一站式平台的个人信息页查看
    3: required string secret_key, // 用户的 secret_key，可以在 SAIL 一站式平台的个人信息页查看
    4: required string appkey, // 应用的 appkey，可以在 SAIL 一站式平台的应用列表页查看
    5: required i64 expiration; // token 自定义过期时间，单位是秒。如果是 auth-v1 这类在线 token，有效期不超过 1 天，否则不能超过 1 年

    51: optional bool check_package_name; // SDK 的参数
    52: optional string package_name; // SDK 的参数
    53: optional list<string> platform; // SDK 的参数

    255: optional base.Base Base;
}

struct GetTokenResponse {
    1: required string task_id, // 本次请求的 id
    2: optional string token = "", // 生成的 token，有效期至 expires_at
    3: optional i64 expires_at = 0, // 过期时间 unix 时间戳，单位是秒

    255: required base.BaseResp BaseResp,
}

service SamiService {
    InvokeResponse Invoke(1: InvokeRequest req)
    GetTokenResponse GetToken(1: GetTokenRequest req)
}
