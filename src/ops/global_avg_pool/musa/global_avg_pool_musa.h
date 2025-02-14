#ifndef __MUSA_GLOBAL_AVG_POOL_H__
#define __MUSA_GLOBAL_AVG_POOL_H__

#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "operators.h"
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <numeric>
#include <vector>

struct GlobalAvgPoolMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t data_size;
    uint64_t y_data_size;
    uint64_t x_per_NC_data_size;
    unsigned max_block_size;
    uint64_t max_grid_size;
    uint64_t items_per_thread;
    std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t;
    musa::dnn::Tensor *x_desc;
    musa::dnn::Tensor *y_desc;
    musa::dnn::Pooling *pool_desc;
    musa::dnn::Tensor *indices;
    const float alpha;
    const float beta;
};

typedef struct GlobalAvgPoolMusaDescriptor *GlobalAvgPoolMusaDescriptor_t;

infiniopStatus_t musaCreateGlobalAvgPoolDescriptor(MusaHandle_t,
                                                   GlobalAvgPoolMusaDescriptor_t *,
                                                   infiniopTensorDescriptor_t y,
                                                   infiniopTensorDescriptor_t x);

infiniopStatus_t musaGetGlobalAvgPoolWorkspaceSize(GlobalAvgPoolMusaDescriptor_t desc, uint64_t *size);

infiniopStatus_t musaGlobalAvgPool(GlobalAvgPoolMusaDescriptor_t desc,
                                   void *workspace, uint64_t workspace_size, void *y, void const *x,
                                   void *stream);

infiniopStatus_t musaDestroyGlobalAvgPoolDescriptor(GlobalAvgPoolMusaDescriptor_t desc);

#endif