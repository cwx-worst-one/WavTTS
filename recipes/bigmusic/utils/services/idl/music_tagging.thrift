include "base.thrift"
namespace cpp lab.speech.music_tagging
namespace py lab.speech.music_tagging
namespace go lab.speech.music_tagging

enum MusicTaggingCode{
    kSuccess = 0,
    kInvalidRequest = 1001,
    kServerBusy=1002,
    kErrorDownload=1003,
    kErrorConvertAudio=1004,
    kErrorExtractFeature=1005,
    kSessionTimeout=1010,
    kErrorTagging = 1016,
    kErrorRpc = 1098,
    kErrorUnknown=1099,
}

struct TaggingRequest {
    1: required string track_id,
    2: required string url,
    3: optional string song_name, 
    4: optional string artist,
    5: optional string album,
    6: optional string lyricist, 
    
    255: optional base.Base Base, 
}

struct TaggingResponse {
    1: required i32 code,
    2: required string result_json,

    255: required base.BaseResp BaseResp,
}


struct EmbeddingResponse {
    1: required i32 code,
    2: required list<double> embedding,

    255: required base.BaseResp BaseResp,
}

service MusicTagging{
    TaggingResponse Tagging(1: TaggingRequest req),
    TaggingResponse TaggingLanguage(1: TaggingRequest req),
    TaggingResponse TaggingGenre(1: TaggingRequest req),
    TaggingResponse TaggingGenre20(1: TaggingRequest req),
    TaggingResponse TaggingMood(1: TaggingRequest req),
    TaggingResponse TaggingTheme(1: TaggingRequest req),
    TaggingResponse TaggingMusicLowQuality(1: TaggingRequest req),
    EmbeddingResponse EmbeddingSSL(1: TaggingRequest req),
}

