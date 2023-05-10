import os
import time
import sys
import sqlite3
import json
import cxxfilt


# from cuda/include/cupti_runtime_cbid.h
RUNTIME_CBID_2_NAME = [
    'Zero',
    'cudaDriverGetVersion_v3020',
    'cudaRuntimeGetVersion_v3020',
    'cudaGetDeviceCount_v3020',
    'cudaGetDeviceProperties_v3020',
    'cudaChooseDevice_v3020',
    'cudaGetChannelDesc_v3020',
    'cudaCreateChannelDesc_v3020',
    'cudaConfigureCall_v3020',
    'cudaSetupArgument_v3020',
    'cudaGetLastError_v3020',
    'cudaPeekAtLastError_v3020',
    'cudaGetErrorString_v3020',
    'cudaLaunch_v3020',
    'cudaFuncSetCacheConfig_v3020',
    'cudaFuncGetAttributes_v3020',
    'cudaSetDevice_v3020',
    'cudaGetDevice_v3020',
    'cudaSetValidDevices_v3020',
    'cudaSetDeviceFlags_v3020',
    'cudaMalloc_v3020',
    'cudaMallocPitch_v3020',
    'cudaFree_v3020',
    'cudaMallocArray_v3020',
    'cudaFreeArray_v3020',
    'cudaMallocHost_v3020',
    'cudaFreeHost_v3020',
    'cudaHostAlloc_v3020',
    'cudaHostGetDevicePointer_v3020',
    'cudaHostGetFlags_v3020',
    'cudaMemGetInfo_v3020',
    'cudaMemcpy_v3020',
    'cudaMemcpy2D_v3020',
    'cudaMemcpyToArray_v3020',
    'cudaMemcpy2DToArray_v3020',
    'cudaMemcpyFromArray_v3020',
    'cudaMemcpy2DFromArray_v3020',
    'cudaMemcpyArrayToArray_v3020',
    'cudaMemcpy2DArrayToArray_v3020',
    'cudaMemcpyToSymbol_v3020',
    'cudaMemcpyFromSymbol_v3020',
    'cudaMemcpyAsync_v3020',
    'cudaMemcpyToArrayAsync_v3020',
    'cudaMemcpyFromArrayAsync_v3020',
    'cudaMemcpy2DAsync_v3020',
    'cudaMemcpy2DToArrayAsync_v3020',
    'cudaMemcpy2DFromArrayAsync_v3020',
    'cudaMemcpyToSymbolAsync_v3020',
    'cudaMemcpyFromSymbolAsync_v3020',
    'cudaMemset_v3020',
    'cudaMemset2D_v3020',
    'cudaMemsetAsync_v3020',
    'cudaMemset2DAsync_v3020',
    'cudaGetSymbolAddress_v3020',
    'cudaGetSymbolSize_v3020',
    'cudaBindTexture_v3020',
    'cudaBindTexture2D_v3020',
    'cudaBindTextureToArray_v3020',
    'cudaUnbindTexture_v3020',
    'cudaGetTextureAlignmentOffset_v3020',
    'cudaGetTextureReference_v3020',
    'cudaBindSurfaceToArray_v3020',
    'cudaGetSurfaceReference_v3020',
    'cudaGLSetGLDevice_v3020',
    'cudaGLRegisterBufferObject_v3020',
    'cudaGLMapBufferObject_v3020',
    'cudaGLUnmapBufferObject_v3020',
    'cudaGLUnregisterBufferObject_v3020',
    'cudaGLSetBufferObjectMapFlags_v3020',
    'cudaGLMapBufferObjectAsync_v3020',
    'cudaGLUnmapBufferObjectAsync_v3020',
    'cudaWGLGetDevice_v3020',
    'cudaGraphicsGLRegisterImage_v3020',
    'cudaGraphicsGLRegisterBuffer_v3020',
    'cudaGraphicsUnregisterResource_v3020',
    'cudaGraphicsResourceSetMapFlags_v3020',
    'cudaGraphicsMapResources_v3020',
    'cudaGraphicsUnmapResources_v3020',
    'cudaGraphicsResourceGetMappedPointer_v3020',
    'cudaGraphicsSubResourceGetMappedArray_v3020',
    'cudaVDPAUGetDevice_v3020',
    'cudaVDPAUSetVDPAUDevice_v3020',
    'cudaGraphicsVDPAURegisterVideoSurface_v3020',
    'cudaGraphicsVDPAURegisterOutputSurface_v3020',
    'cudaD3D11GetDevice_v3020',
    'cudaD3D11GetDevices_v3020',
    'cudaD3D11SetDirect3DDevice_v3020',
    'cudaGraphicsD3D11RegisterResource_v3020',
    'cudaD3D10GetDevice_v3020',
    'cudaD3D10GetDevices_v3020',
    'cudaD3D10SetDirect3DDevice_v3020',
    'cudaGraphicsD3D10RegisterResource_v3020',
    'cudaD3D10RegisterResource_v3020',
    'cudaD3D10UnregisterResource_v3020',
    'cudaD3D10MapResources_v3020',
    'cudaD3D10UnmapResources_v3020',
    'cudaD3D10ResourceSetMapFlags_v3020',
    'cudaD3D10ResourceGetSurfaceDimensions_v3020',
    'cudaD3D10ResourceGetMappedArray_v3020',
    'cudaD3D10ResourceGetMappedPointer_v3020',
    'cudaD3D10ResourceGetMappedSize_v3020',
    'cudaD3D10ResourceGetMappedPitch_v3020',
    'cudaD3D9GetDevice_v3020',
    'cudaD3D9GetDevices_v3020',
    'cudaD3D9SetDirect3DDevice_v3020',
    'cudaD3D9GetDirect3DDevice_v3020',
    'cudaGraphicsD3D9RegisterResource_v3020',
    'cudaD3D9RegisterResource_v3020',
    'cudaD3D9UnregisterResource_v3020',
    'cudaD3D9MapResources_v3020',
    'cudaD3D9UnmapResources_v3020',
    'cudaD3D9ResourceSetMapFlags_v3020',
    'cudaD3D9ResourceGetSurfaceDimensions_v3020',
    'cudaD3D9ResourceGetMappedArray_v3020',
    'cudaD3D9ResourceGetMappedPointer_v3020',
    'cudaD3D9ResourceGetMappedSize_v3020',
    'cudaD3D9ResourceGetMappedPitch_v3020',
    'cudaD3D9Begin_v3020',
    'cudaD3D9End_v3020',
    'cudaD3D9RegisterVertexBuffer_v3020',
    'cudaD3D9UnregisterVertexBuffer_v3020',
    'cudaD3D9MapVertexBuffer_v3020',
    'cudaD3D9UnmapVertexBuffer_v3020',
    'cudaThreadExit_v3020',
    'cudaSetDoubleForDevice_v3020',
    'cudaSetDoubleForHost_v3020',
    'cudaThreadSynchronize_v3020',
    'cudaThreadGetLimit_v3020',
    'cudaThreadSetLimit_v3020',
    'cudaStreamCreate_v3020',
    'cudaStreamDestroy_v3020',
    'cudaStreamSynchronize_v3020',
    'cudaStreamQuery_v3020',
    'cudaEventCreate_v3020',
    'cudaEventCreateWithFlags_v3020',
    'cudaEventRecord_v3020',
    'cudaEventDestroy_v3020',
    'cudaEventSynchronize_v3020',
    'cudaEventQuery_v3020',
    'cudaEventElapsedTime_v3020',
    'cudaMalloc3D_v3020',
    'cudaMalloc3DArray_v3020',
    'cudaMemset3D_v3020',
    'cudaMemset3DAsync_v3020',
    'cudaMemcpy3D_v3020',
    'cudaMemcpy3DAsync_v3020',
    'cudaThreadSetCacheConfig_v3020',
    'cudaStreamWaitEvent_v3020',
    'cudaD3D11GetDirect3DDevice_v3020',
    'cudaD3D10GetDirect3DDevice_v3020',
    'cudaThreadGetCacheConfig_v3020',
    'cudaPointerGetAttributes_v4000',
    'cudaHostRegister_v4000',
    'cudaHostUnregister_v4000',
    'cudaDeviceCanAccessPeer_v4000',
    'cudaDeviceEnablePeerAccess_v4000',
    'cudaDeviceDisablePeerAccess_v4000',
    'cudaPeerRegister_v4000',
    'cudaPeerUnregister_v4000',
    'cudaPeerGetDevicePointer_v4000',
    'cudaMemcpyPeer_v4000',
    'cudaMemcpyPeerAsync_v4000',
    'cudaMemcpy3DPeer_v4000',
    'cudaMemcpy3DPeerAsync_v4000',
    'cudaDeviceReset_v3020',
    'cudaDeviceSynchronize_v3020',
    'cudaDeviceGetLimit_v3020',
    'cudaDeviceSetLimit_v3020',
    'cudaDeviceGetCacheConfig_v3020',
    'cudaDeviceSetCacheConfig_v3020',
    'cudaProfilerInitialize_v4000',
    'cudaProfilerStart_v4000',
    'cudaProfilerStop_v4000',
    'cudaDeviceGetByPCIBusId_v4010',
    'cudaDeviceGetPCIBusId_v4010',
    'cudaGLGetDevices_v4010',
    'cudaIpcGetEventHandle_v4010',
    'cudaIpcOpenEventHandle_v4010',
    'cudaIpcGetMemHandle_v4010',
    'cudaIpcOpenMemHandle_v4010',
    'cudaIpcCloseMemHandle_v4010',
    'cudaArrayGetInfo_v4010',
    'cudaFuncSetSharedMemConfig_v4020',
    'cudaDeviceGetSharedMemConfig_v4020',
    'cudaDeviceSetSharedMemConfig_v4020',
    'cudaCreateTextureObject_v5000',
    'cudaDestroyTextureObject_v5000',
    'cudaGetTextureObjectResourceDesc_v5000',
    'cudaGetTextureObjectTextureDesc_v5000',
    'cudaCreateSurfaceObject_v5000',
    'cudaDestroySurfaceObject_v5000',
    'cudaGetSurfaceObjectResourceDesc_v5000',
    'cudaMallocMipmappedArray_v5000',
    'cudaGetMipmappedArrayLevel_v5000',
    'cudaFreeMipmappedArray_v5000',
    'cudaBindTextureToMipmappedArray_v5000',
    'cudaGraphicsResourceGetMappedMipmappedArray_v5000',
    'cudaStreamAddCallback_v5000',
    'cudaStreamCreateWithFlags_v5000',
    'cudaGetTextureObjectResourceViewDesc_v5000',
    'cudaDeviceGetAttribute_v5000',
    'cudaStreamDestroy_v5050',
    'cudaStreamCreateWithPriority_v5050',
    'cudaStreamGetPriority_v5050',
    'cudaStreamGetFlags_v5050',
    'cudaDeviceGetStreamPriorityRange_v5050',
    'cudaMallocManaged_v6000',
    'cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6000',
    'cudaStreamAttachMemAsync_v6000',
    'cudaGetErrorName_v6050',
    'cudaOccupancyMaxActiveBlocksPerMultiprocessor_v6050',
    'cudaLaunchKernel_v7000',
    'cudaGetDeviceFlags_v7000',
    'cudaLaunch_ptsz_v7000',
    'cudaLaunchKernel_ptsz_v7000',
    'cudaMemcpy_ptds_v7000',
    'cudaMemcpy2D_ptds_v7000',
    'cudaMemcpyToArray_ptds_v7000',
    'cudaMemcpy2DToArray_ptds_v7000',
    'cudaMemcpyFromArray_ptds_v7000',
    'cudaMemcpy2DFromArray_ptds_v7000',
    'cudaMemcpyArrayToArray_ptds_v7000',
    'cudaMemcpy2DArrayToArray_ptds_v7000',
    'cudaMemcpyToSymbol_ptds_v7000',
    'cudaMemcpyFromSymbol_ptds_v7000',
    'cudaMemcpyAsync_ptsz_v7000',
    'cudaMemcpyToArrayAsync_ptsz_v7000',
    'cudaMemcpyFromArrayAsync_ptsz_v7000',
    'cudaMemcpy2DAsync_ptsz_v7000',
    'cudaMemcpy2DToArrayAsync_ptsz_v7000',
    'cudaMemcpy2DFromArrayAsync_ptsz_v7000',
    'cudaMemcpyToSymbolAsync_ptsz_v7000',
    'cudaMemcpyFromSymbolAsync_ptsz_v7000',
    'cudaMemset_ptds_v7000',
    'cudaMemset2D_ptds_v7000',
    'cudaMemsetAsync_ptsz_v7000',
    'cudaMemset2DAsync_ptsz_v7000',
    'cudaStreamGetPriority_ptsz_v7000',
    'cudaStreamGetFlags_ptsz_v7000',
    'cudaStreamSynchronize_ptsz_v7000',
    'cudaStreamQuery_ptsz_v7000',
    'cudaStreamAttachMemAsync_ptsz_v7000',
    'cudaEventRecord_ptsz_v7000',
    'cudaMemset3D_ptds_v7000',
    'cudaMemset3DAsync_ptsz_v7000',
    'cudaMemcpy3D_ptds_v7000',
    'cudaMemcpy3DAsync_ptsz_v7000',
    'cudaStreamWaitEvent_ptsz_v7000',
    'cudaStreamAddCallback_ptsz_v7000',
    'cudaMemcpy3DPeer_ptds_v7000',
    'cudaMemcpy3DPeerAsync_ptsz_v7000',
    'cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000',
    'cudaMemPrefetchAsync_v8000',
    'cudaMemPrefetchAsync_ptsz_v8000',
    'cudaMemAdvise_v8000',
    'cudaDeviceGetP2PAttribute_v8000',
    'cudaGraphicsEGLRegisterImage_v7000',
    'cudaEGLStreamConsumerConnect_v7000',
    'cudaEGLStreamConsumerDisconnect_v7000',
    'cudaEGLStreamConsumerAcquireFrame_v7000',
    'cudaEGLStreamConsumerReleaseFrame_v7000',
    'cudaEGLStreamProducerConnect_v7000',
    'cudaEGLStreamProducerDisconnect_v7000',
    'cudaEGLStreamProducerPresentFrame_v7000',
    'cudaEGLStreamProducerReturnFrame_v7000',
    'cudaGraphicsResourceGetMappedEglFrame_v7000',
    'cudaMemRangeGetAttribute_v8000',
    'cudaMemRangeGetAttributes_v8000',
    'cudaEGLStreamConsumerConnectWithFlags_v7000',
    'cudaLaunchCooperativeKernel_v9000',
    'cudaLaunchCooperativeKernel_ptsz_v9000',
    'cudaEventCreateFromEGLSync_v9000',
    'cudaLaunchCooperativeKernelMultiDevice_v9000',
    'cudaFuncSetAttribute_v9000',
    'cudaImportExternalMemory_v10000',
    'cudaExternalMemoryGetMappedBuffer_v10000',
    'cudaExternalMemoryGetMappedMipmappedArray_v10000',
    'cudaDestroyExternalMemory_v10000',
    'cudaImportExternalSemaphore_v10000',
    'cudaSignalExternalSemaphoresAsync_v10000',
    'cudaSignalExternalSemaphoresAsync_ptsz_v10000',
    'cudaWaitExternalSemaphoresAsync_v10000',
    'cudaWaitExternalSemaphoresAsync_ptsz_v10000',
    'cudaDestroyExternalSemaphore_v10000',
    'cudaLaunchHostFunc_v10000',
    'cudaLaunchHostFunc_ptsz_v10000',
    'cudaGraphCreate_v10000',
    'cudaGraphKernelNodeGetParams_v10000',
    'cudaGraphKernelNodeSetParams_v10000',
    'cudaGraphAddKernelNode_v10000',
    'cudaGraphAddMemcpyNode_v10000',
    'cudaGraphMemcpyNodeGetParams_v10000',
    'cudaGraphMemcpyNodeSetParams_v10000',
    'cudaGraphAddMemsetNode_v10000',
    'cudaGraphMemsetNodeGetParams_v10000',
    'cudaGraphMemsetNodeSetParams_v10000',
    'cudaGraphAddHostNode_v10000',
    'cudaGraphHostNodeGetParams_v10000',
    'cudaGraphAddChildGraphNode_v10000',
    'cudaGraphChildGraphNodeGetGraph_v10000',
    'cudaGraphAddEmptyNode_v10000',
    'cudaGraphClone_v10000',
    'cudaGraphNodeFindInClone_v10000',
    'cudaGraphNodeGetType_v10000',
    'cudaGraphGetRootNodes_v10000',
    'cudaGraphNodeGetDependencies_v10000',
    'cudaGraphNodeGetDependentNodes_v10000',
    'cudaGraphAddDependencies_v10000',
    'cudaGraphRemoveDependencies_v10000',
    'cudaGraphDestroyNode_v10000',
    'cudaGraphInstantiate_v10000',
    'cudaGraphLaunch_v10000',
    'cudaGraphLaunch_ptsz_v10000',
    'cudaGraphExecDestroy_v10000',
    'cudaGraphDestroy_v10000',
    'cudaStreamBeginCapture_v10000',
    'cudaStreamBeginCapture_ptsz_v10000',
    'cudaStreamIsCapturing_v10000',
    'cudaStreamIsCapturing_ptsz_v10000',
    'cudaStreamEndCapture_v10000',
    'cudaStreamEndCapture_ptsz_v10000',
    'cudaGraphHostNodeSetParams_v10000',
    'cudaGraphGetNodes_v10000',
    'cudaGraphGetEdges_v10000',
    'cudaStreamGetCaptureInfo_v10010',
    'cudaStreamGetCaptureInfo_ptsz_v10010',
    'cudaGraphExecKernelNodeSetParams_v10010',
    'cudaThreadExchangeStreamCaptureMode_v10010',
    'cudaDeviceGetNvSciSyncAttributes_v10020',
    'cudaOccupancyAvailableDynamicSMemPerBlock_v10200',
    'cudaStreamSetFlags_v10200',
    'cudaStreamSetFlags_ptsz_v10200',
    'cudaGraphExecMemcpyNodeSetParams_v10020',
    'cudaGraphExecMemsetNodeSetParams_v10020',
    'cudaGraphExecHostNodeSetParams_v10020',
    'cudaGraphExecUpdate_v10020',
]


def get_desc_info(conn):
    desc = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        try:
            desc[r["id"]] = cxxfilt.demangle(r["value"])
        except Exception:
            desc[r["id"]] = r["value"]
    return desc

def get_mark_info(conn, pid, desc):
    marker_query = """
     SELECT
         start.id AS marker_id, start.name, start.timestamp AS start_time, end.timestamp AS end_time
     FROM
         CUPTI_ACTIVITY_KIND_MARKER AS start INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
         ON start.id = end.id
     WHERE
         start.name != 0 AND end.name = 0
     """
    res = []
    for row in conn.execute(marker_query):
        event = {
                "name": desc[row["name"]],
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "CPU",
                "ts": row["start_time"] / 1000,
                "dur": (row["end_time"] - row["start_time"]) / 1000,
                "tid": "Marker",
                "pid": pid,
                "args": {
                    "id": row["marker_id"],
                    },
                }
        res.append(event)
    return res

def get_runtime_info(conn, pid, desc):
    res = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        cbid = row['cbid']
        if cbid not in (20, 22, 131, 147):
            continue
        event = {
                "name": RUNTIME_CBID_2_NAME[cbid],
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "CPU",
                "ts": row["start"] / 1000,
                "dur": (row["end"] - row["start"]) / 1000,
                "tid": "runtime",
                "pid": pid,
                "args": {
                    "correlationId": row["correlationId"],
                    },
                }
        res.append(event)
    return res

def get_compute_info(conn, pid, desc):
    res = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        event = {
                "name": desc[row["name"]],
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "cuda",
                "ts": row["start"] / 1000,
                "dur": (row["end"] - row["start"]) / 1000,
                "tid": "Computation",
                # TODO: lookup GPU name
                "pid": pid,
                "args": {
                    "Stream": row["streamId"],
                    "Grid size": "[ {}, {}, {} ]".format(row["gridX"], row["gridY"], row["gridZ"]),
                    "Block size": "[ {}, {}, {} ]".format(row["blockX"], row["blockY"], row["blockZ"]),
                    },
                }
        res.append(event)
    return res

def readable_size(size):
    i = 0
    dw = ['B', 'KB', 'MB', 'GB', 'TB']
    while abs(size) >= 1024.0 and i + 1 < len(dw):
        size /= 1024
        i += 1
    return '{:.3f} {}'.format(size, dw[i])

def get_mcp_info(conn, pid, _desc):
    res = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
        if row["copyKind"] == 1:
            copyKind = "HtoD"
        elif row["copyKind"] == 2:
            copyKind = "DtoH"
        elif row["copyKind"] == 8:
            copyKind = "DtoD"
        else:
            copyKind = str(row["copyKind"])
        if row["flags"] == 0:
            flags = "sync"
        elif row["flags"] == 1:
            flags = "async"
        else:
            flags = str(row["flags"])
        event = {
                "name": "Memcpy {} [{}]".format(copyKind, flags),
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "cuda",
                "ts": row["start"] / 1000,
                "dur": (row["end"] - row["start"]) / 1000,
                "tid": "Memory",
                # TODO: lookup GPU name.  This is stored in
                # CUPTI_ACTIVITY_KIND_DEVICE
                "pid": pid,
                "args": {
                    "Size": readable_size(row["bytes"]),
                    # TODO: More
                    },
                }
        res.append(event)
    return res

def get_mset_info(conn, pid, _desc):
    res = []
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMSET"):
        if row["flags"] == 0:
            flags = "sync"
        elif row["flags"] == 1:
            flags = "async"
        else:
            flags = str(row["flags"])
        memoryKind = row['memoryKind']
        event = {
                "name": "Memset {} [{}]".format(memoryKind, flags),
                "ph": "X", # Complete Event (Begin + End event)
                "cat": "cuda",
                "ts": row["start"] / 1000,
                "dur": (row["end"] - row["start"]) / 1000,
                "tid": "Memory",
                "pid": pid,
                "args": {
                    "Size": readable_size(row["bytes"]),
                    # TODO: More
                    },
                }
        res.append(event)
    return res

def get_filtered_compute_info(conn, pid, desc):
    res = []
    for r in get_compute_info(conn, pid, desc):
        name = r['name'].lower()
        if 'nccl' not in name and 'dolphin' not in name:
            continue
        res.append(r)
    return res


def get_filterd_runtime_info(conn, pid, desc):
    res = []
    for r in get_runtime_info(conn, pid, desc):
        name = r['name'].lower()
        if 'alloc' not in name and 'free' not in name:
            continue
        res.append(r)
    return res

INFO_FNS = [get_filtered_compute_info, get_filterd_runtime_info]

def main():
    trace_file = sys.argv[1] + '.json'

    res = []
    for pid in sys.argv[1:]:
        conn = sqlite3.connect(pid)
        conn.row_factory = sqlite3.Row
        desc = get_desc_info(conn)
        for fn in INFO_FNS:
            try:
                res += fn(conn, pid, desc)
            except Exception:
                pass
        conn.close()

    with open(trace_file, 'w') as f:
        f.write(json.dumps(res))

if __name__ == '__main__':
    main()
