namespace go lab.mt.langdetect
namespace py lab.mt.langdetect
namespace rs lab.mt.langdetect
namespace cpp langdetect

include "./base.thrift"

struct ResponseInfo{
    1: required string DetectText
    2: required string LangCode = "un"   // 检测的语种(mt.thrift.Language)
    3: optional string LangName = "未知"
    4: required double Probability = 0   // 语种检测对应的置信度
}

struct LangDetectRequest{
    1: required string DetectText          // 必填, 进行语种检测的文本
    2: optional string Business = "test"   // 选填, 服务请求方业务名称
    255: optional base.Base Base           // 将请求方 P.S.M 填写到 Caller 中
}
struct LangDetectResponse{
    1: required ResponseInfo Responseinfo
    255: optional base.BaseResp BaseResp
}

struct LangDetectRequestList{
    1: required list<string> DetectTextList  // 待识别的文本列表
    2: optional string Business = "test"     // 选填, 服务请求方业务名称
    255: optional base.Base Base
}
struct LangDetectResponseList{
    1: required list<ResponseInfo> ResponseinfoList
    255: optional base.BaseResp BaseResp
}

service LabMTLangdetect {
    LangDetectResponse detect(1:LangDetectRequest req);
    LangDetectResponseList detect_batch(1:LangDetectRequestList req);
}

