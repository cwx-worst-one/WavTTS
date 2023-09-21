include "base.thrift"

namespace go lab.sami
namespace py lab.sami


struct InvokeRequest {
    1: required string access_key,      // Use "sami_audio_fetch"
    2: required string method,          // Use "GeneralAudioFetch"
    3: optional string payload,         // See bellow for detailed instruction
    4: optional binary data,            // Omit
    5: required string version = "v4",  // Use "v4"

    6: optional string task_id = "",    // Use ""
    7: optional bool is_offline = false,// Use false

    255: required base.Base Base,
}

struct InvokeResponse {
    1: required string task_id,         // Omit
    2: optional string payload,         // See bellow for detailed instruction
    3: optional binary data,            // See bellow for detailed instruction

    255: required base.BaseResp BaseResp,
}

service SamiService {
    InvokeResponse Invoke(1: InvokeRequest req)
}
