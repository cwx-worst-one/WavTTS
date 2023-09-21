namespace go lab.sami.app
namespace py lab.sami.app

struct MetaData {
    1: required string access_key,
    2: required string user_id,
    3: required string task_id,
    4: required string message_id
    5: required string model_id,
    6: required string event_name,
    7: required string version = "v1",
}

struct ChargeData {
    1: required string type,
    2: required i32 amount,
}

struct InvokeRequest {
    1: required MetaData meta_data,
    2: optional binary data,
    3: optional string payload,
}

struct InvokeResponse {
    1: required MetaData meta_data,
    2: required i32 status,
    3: optional binary data,
    4: optional string payload,
    5: optional ChargeData charge_data,
    6: optional i32 status_code,
    7: optional string status_text,
}

service Service {
    InvokeResponse Invoke(1: InvokeRequest req)
}
