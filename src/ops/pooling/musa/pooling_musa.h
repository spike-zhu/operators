#ifndef __MUSA_POOLING_H__
#define __MUSA_POOLING_H__

#include "../../../devices/musa/musa_handle.h"
#include "operators.h"

struct PoolingMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t;
    musa::dnn::Tensor* x_tensor;
    musa::dnn::Tensor* y_tensor;
    musa::dnn::Tensor* indices_tensor;
    musa::dnn::Pooling* pool_operator;
    const float alpha;
    const float beta;
};

typedef struct PoolingMusaDescriptor *PoolingMusaDescriptor_t;

infiniopStatus_t musaCreatePoolingDescriptor(MusaHandle_t handle,
                                             PoolingMusaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             uint64_t const *kernel_shape,
                                             uint64_t const *pads,
                                             int64_t const *strides,
                                             uint64_t n,
                                             int pooling_type);

infiniopStatus_t musaGetPoolingWorkspaceSize(PoolingMusaDescriptor_t desc, uint64_t *size);

infiniopStatus_t musaPooling(PoolingMusaDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *y,
                             void const *x,
                             void *stream);

infiniopStatus_t musaDestroyPoolingDescriptor(PoolingMusaDescriptor_t desc);

inline musa::dnn::Pooling::Mode getPoolingMode(int pooling_type) {
    switch (pooling_type) {
        case 0:
            return musa::dnn::Pooling::Mode::MAXPOOL;
        case 1:
            return musa::dnn::Pooling::Mode::AVGPOOL_COUNT_PAD;
            // return musa::dnn::Pooling::Mode::AVGPOOL_COUNT_WITHOUT_PAD;
        default:
            return musa::dnn::Pooling::Mode::MAXPOOL;
    }
}

#endif// __CUDA_POOLING_H__
