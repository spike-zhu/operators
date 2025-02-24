#include "../utils.h"
#include "operators.h"
#include "ops/concat/concat.h"

#ifdef ENABLE_CPU
#include "cpu/concat_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/concat.cuh"
#endif
#ifdef ENABLE_MT_GPU
#include "../../devices/musa/musa_handle.h"
#include "musa/concat_musa.h"
#endif

__C infiniopStatus_t infiniopCreateConcatDescriptor(
    infiniopHandle_t handle,
    infiniopConcatDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t *x,
    uint64_t num_inputs,
    int64_t axis) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateConcatDescriptor(handle, (ConcatCpuDescriptor_t *) desc_ptr, y, x, num_inputs, axis);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateConcatDescriptor((CudaHandle_t) handle, (ConcatCudaDescriptor_t *) desc_ptr, y, x, num_inputs, axis);
        }
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu:{
            return musaCreateConcatDescriptor((MusaHandle_t) handle, (ConcatMusaDescriptor_t *) desc_ptr, y, x, num_inputs, axis);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}


__C infiniopStatus_t infiniopConcat(infiniopConcatDescriptor_t desc, void *y, void const **x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuConcat((ConcatCpuDescriptor_t) desc, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            printf("[INTO ENABLE_NV_GPU]\n");
            return cudaConcat((ConcatCudaDescriptor_t) desc, y, x, stream);
        }

#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu: {

            return musaConcat((ConcatMusaDescriptor_t) desc, y, x, stream);
        }
#endif

    }
    return STATUS_BAD_TENSOR_SHAPE;
}


__C infiniopStatus_t infiniopDestroyConcatDescriptor(infiniopConcatDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyConcatDescriptor((ConcatCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyConcatDescriptor((ConcatCudaDescriptor_t) desc);
        }
#endif
#ifdef ENABLE_MT_GPU
        case DevMtGpu: {
            return musaDestroyConcatDescriptor((ConcatMusaDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
