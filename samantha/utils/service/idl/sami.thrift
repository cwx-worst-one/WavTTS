include "base.thrift"

namespace go lab.sami
namespace py lab.sami

struct InvokeRequest {
    1: required string access_key,      // user access key
    2: required string method,          // method namespace
    3: optional string payload,         // json payload
    4: optional binary data,            // binary data
    5: required string version = "v4",  // version

    6: optional string task_id = "",    // task id
    7: optional bool is_offline = false,// is offline

    255: required base.Base Base,
}

struct InvokeResponse {
    1: required string task_id,         // task id
    2: optional string payload,         // returned payload
    3: optional binary data,            // returned binary data
    4: optional string state,           // state

    255: required base.BaseResp BaseResp,
}

service SamiService {
    InvokeResponse Invoke(1: InvokeRequest req)
}

service SAMI {
    InvokeResponse Invoke(1: InvokeRequest req)
}
