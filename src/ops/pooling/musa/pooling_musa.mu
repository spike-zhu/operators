#include "../../../devices/musa/common_musa.h"
#include "pooling_musa.h"
#include <vector>

infiniopStatus_t pooling_mt_gpu(PoolingMusaDescriptor_t desc, void *y, void const *x, void *stream) {
    checkMusaError(musaSetDevice(desc->device_id));

    desc->x_tensor->SetAddr(x);
    desc->y_tensor->SetAddr(y);

    use_mudnn(desc->mudnn_handles_t, desc->device_id, (musaStream_t) stream, [&](musa::dnn::Handle* handle) {
        desc->pool_operator->Run(*handle, *(desc->y_tensor), *(desc->x_tensor), *(desc->indices_tensor));
    });
    
    return STATUS_SUCCESS;
}

infiniopStatus_t musaPooling(PoolingMusaDescriptor_t desc,
                             void *workspace, uint64_t workspace_size,
                             void *y, void const *x, void *stream) {
    if (desc->dtype == F16 || desc->dtype == F32) {
        return pooling_mt_gpu(desc, y, x, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
