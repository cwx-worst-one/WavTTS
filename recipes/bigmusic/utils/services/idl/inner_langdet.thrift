namespace go inner_langdet
namespace py inner_langdet
namespace rs inner_langdet
namespace cpp inner_langdet

include "./base.thrift"

struct Item{
    1: required string LangCode = "un",
    2: optional string LangName = "未知",
    3: required double Probability = 0,
}
struct Response{
    1: required list<Item> ans,
    255: optional base.BaseResp BaseResp
}

struct Request{
    1: required list<string> DetectTextList,
    255: optional base.Base Base
}

struct LangidRes{
    1: required list<double> ans,
    255: optional base.BaseResp BaseResp
}

struct LangidReq{
    1: required string text,
    255: optional base.Base Base
}

struct GenrePostReq{
    1: required list<double> probs,
    255: optional base.Base Base
}

struct GenrePostRes{
    1: required string result,
    255: optional base.Base Base
}

struct MoodPostReq{
    1: required list<double> probs,
    255: optional base.Base Base
}

struct MoodPostRes{
    1: required string result,
    255: optional base.Base Base
}

struct ThemePostReq{
    1: required list<double> probs,
    255: optional base.Base Base
}

struct ThemePostRes{
    1: required string result,
    255: optional base.Base Base
}

service InnerLangdet {
    Response detect(1:Request req);
    LangidRes langidv1(1:LangidReq req);
    LangidRes langidv2(1:LangidReq req);
    GenrePostRes genre20_post(1:GenrePostReq req);
    MoodPostRes mood_post(1:MoodPostReq req);
    ThemePostRes theme_post(1:ThemePostReq req);
}

