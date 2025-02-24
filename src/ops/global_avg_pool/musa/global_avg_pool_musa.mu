#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"
#include "global_avg_pool_musa.h"


infiniopStatus_t global_avg_pool_mt_gpu(GlobalAvgPoolMusaDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream, unsigned pack_size) {
    // use muDNN lib

    checkMusaError(musaSetDevice(desc->device_id));
    desc->y_desc->SetAddr(y);
    desc->x_desc->SetAddr(x);

    use_mudnn(desc->mudnn_handles_t, desc->device_id, (musaStream_t) stream, [&](musa::dnn::Handle* handle) {
        desc->pool_desc->Run(*handle, *(desc->y_desc), *(desc->x_desc), *(desc->indices));
    });

    printf("[SUCCESS to execute global_avg_pool_mt_gpu]\n");

    return STATUS_SUCCESS;
}



infiniopStatus_t musaGlobalAvgPool(GlobalAvgPoolMusaDescriptor_t desc,
                                   void *workspace, uint64_t workspace_size,
                                   void *y, void const *x,
                                   void *stream) {
    checkMusaError(musaSetDevice(desc->device_id));

    if (desc->dtype == F32) {
        return global_avg_pool_mt_gpu(desc, workspace, workspace_size, y, x, stream, 1);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
