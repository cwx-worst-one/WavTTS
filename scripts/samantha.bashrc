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
