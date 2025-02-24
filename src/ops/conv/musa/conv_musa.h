#ifndef __MUSA_CONV_H__
#define __MUSA_CONV_H__

#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "operators.h"
#include <mudnn.h>

struct ConvMusaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    std::shared_ptr<Pool<musa::dnn::Handle>> mudnn_handles_t;
    musa::dnn::Tensor* x_tensor;
    musa::dnn::Tensor* w_tensor;
    musa::dnn::Tensor* y_tensor;
    musa::dnn::Convolution* conv_operator;
    musa::dnn::Convolution::Algorithm algo;
    const float alpha;
    const float beta;
    uint64_t workspace_size;
    musa::dnn::MemoryMaintainer maintainer;
};

typedef struct ConvMusaDescriptor *ConvMusaDescriptor_t;

infiniopStatus_t musaCreateConvDescriptor(MusaHandle_t,
                                          ConvMusaDescriptor_t *,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t w,
                                          void const *pads,
                                          void const *strides,
                                          void const *dilations,
                                          uint64_t n);

infiniopStatus_t musaGetConvWorkspaceSize(ConvMusaDescriptor_t desc, uint64_t *size);

infiniopStatus_t musaConv(ConvMusaDescriptor_t desc,
                          void *workspace, uint64_t workspace_size,
                          void *y, void const *x, void const *w,
                          void *stream);

infiniopStatus_t musaDestroyConvDescriptor(ConvMusaDescriptor_t desc);

#endif
