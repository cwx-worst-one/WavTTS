quota_pool=${ARNOLD_QUOTA_POOL:=default}
if [ "${quota_pool}" == "third_party_aws" ]; then
  export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
fi
unset quota_pool

system_cuda_lib_version=$(realpath /usr/lib/x86_64-linux-gnu/libcuda.so | sed 's=/usr/lib/x86_64-linux-gnu/libcuda.so.\([^.]*\).*=\1=')
compat_cuda_lib_version=$(realpath /usr/local/cuda/compat/libcuda.so | sed 's=/usr/local/cuda.*/libcuda.so.\([^.]*\).*=\1=')
if [ "${system_cuda_lib_version}" -lt "${compat_cuda_lib_version}" ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
fi
unset system_cuda_lib_version compat_cuda_lib_version

MEM_LIMIT=${MY_MEM_LIMIT:=0}
# 1.8T
if [ ${MEM_LIMIT} -ge 1979120929997 ]; then
    export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=100000000
else
    export TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=100000
fi
