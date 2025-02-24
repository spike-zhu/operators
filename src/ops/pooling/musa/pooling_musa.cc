#include "pooling_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"
#include <numeric>

infiniopStatus_t musaCreatePoolingDescriptor(MusaHandle_t handle,
                                             PoolingMusaDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t y,
                                             infiniopTensorDescriptor_t x,
                                             uint64_t const *kernel_shape,
                                             uint64_t const *pads,
                                             int64_t const *strides,
                                             uint64_t n,
                                             int pooling_type) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != n + 2) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || x->shape[1] != y->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (pooling_type > 1) {
        return STATUS_BAD_PARAM;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    const uint64_t new_ndim = ndim;

    int64_t *x_shape = new int64_t[new_ndim];
    int64_t *y_shape = new int64_t[new_ndim];
    for(size_t i = 0; i < new_ndim; ++i){
        x_shape[i] = static_cast<int64_t>(x->shape[i]);
        y_shape[i] = static_cast<int64_t>(y->shape[i]);

    }

    musa::dnn::Tensor *x_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *y_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *indices = new musa::dnn::Tensor(); 

    x_tensor->SetNdInfo((int)new_ndim, x_shape);
    y_tensor->SetNdInfo((int)new_ndim, y_shape);
    indices->SetNdInfo((int)new_ndim, y_shape);

    if (y->dt == F16) {
        x_tensor->SetType(musa::dnn::Tensor::Type::HALF);
        y_tensor->SetType(musa::dnn::Tensor::Type::HALF);
        indices->SetType(musa::dnn::Tensor::Type::HALF);
    } else if (y->dt == F32) {
        x_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
        y_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
        indices->SetType(musa::dnn::Tensor::Type::FLOAT);
    }

    if (new_ndim == 5) {
        x_tensor->SetFormat(musa::dnn::Tensor::Format::NCDHW);
        y_tensor->SetFormat(musa::dnn::Tensor::Format::NCDHW);
        indices->SetFormat(musa::dnn::Tensor::Format::NCDHW);
    }
    else if (new_ndim == 4) {
        x_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
        y_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
        indices->SetFormat(musa::dnn::Tensor::Format::NCHW);
    }
    else if (new_ndim == 3) {
        x_tensor->SetFormat(musa::dnn::Tensor::Format::NCW);
        y_tensor->SetFormat(musa::dnn::Tensor::Format::NCW);
        indices->SetFormat(musa::dnn::Tensor::Format::NCW);
    }
    else {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    musa::dnn::Status status;
    musa::dnn::Pooling* pooling_operator = new musa::dnn::Pooling();

    status = pooling_operator->SetMode(getPoolingMode(pooling_type));
    // if (status == musa::dnn::Status::SUCCESS) {
    //     printf("pool_desc SetMode status:%d\n", static_cast<int>(status));
    // }

    std::initializer_list<int> kernel = {static_cast<int>(kernel_shape[0]), static_cast<int>(kernel_shape[1])};
    std::initializer_list<int> pad = {static_cast<int>(pads[0]), static_cast<int>(pads[1])};
    std::initializer_list<int> stride = {static_cast<int>(strides[0]), static_cast<int>(strides[1])};
    std::initializer_list<int> dilationList = {1, 1};

    status = pooling_operator->SetNdInfo(kernel, pad, stride, dilationList);
    // if (status == musa::dnn::Status::SUCCESS) {
    //     printf("pool_desc SetNdInfo status:%d\n", static_cast<int>(status));
    // }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    *desc_ptr = new PoolingMusaDescriptor{
        DevMtGpu,
        y->dt,
        handle->device_id,
        handle->mudnn_handles_t,
        x_tensor,
        y_tensor,
        indices,
        pooling_operator,
        alpha,
        beta,
    };

    delete[] x_shape;
    delete[] y_shape;
    
    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetPoolingWorkspaceSize(PoolingMusaDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyPoolingDescriptor(PoolingMusaDescriptor_t desc) {
    delete(desc->x_tensor);
    delete(desc->y_tensor);
    delete(desc->indices_tensor);
    delete(desc->pool_operator);

    desc->mudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
