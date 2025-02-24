#include "global_avg_pool_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"

infiniopStatus_t musaCreateGlobalAvgPoolDescriptor(MusaHandle_t handle,
                                                   GlobalAvgPoolMusaDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y,
                                                   infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (ndim <= 2 || ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    for (size_t i = 0; i < ndim; ++i) {
        if (i < 2 && y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        } else if (i >= 2 && y->shape[i] != 1) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
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
    indices->SetNdInfo((int)new_ndim, x_shape);

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


    int N = x_shape[0];  // batch size
    int C = x_shape[1];  // channels
    int D = x_shape[2];
    int H = x_shape[3];  // height
    int W = x_shape[4];  // width

    int kernel[] = {D, H, W};       
    int pad[] = {0, 0, 0};          
    int stride[] = {D, H, W};       
    int dilation[] = {1, 1, 1};     
        
    musa::dnn::Status status;
    musa::dnn::Pooling *pool_desc = new musa::dnn::Pooling();

    status = pool_desc->SetMode(musa::dnn::Pooling::Mode::GLOBAL_AVGPOOL);
    // if (status == musa::dnn::Status::SUCCESS) {
    //     printf("pool_desc SetMode status:%d\n", static_cast<int>(status));
    // }

    // status = pool_desc->SetNdInfo(3, kernel, pad, stride, dilation);
    // if (status == musa::dnn::Status::SUCCESS) {
    //     printf("pool_desc SetNdInfo status:%d\n", static_cast<int>(status));
    // }

    // status = pool_desc->SetNdInfo({H}, {0}, {H}, {1});
    // if (status == musa::dnn::Status::SUCCESS) {
    //     printf("pool_desc SetNdInfo status:%d\n", static_cast<int>(status));
    // }

    // status = pool_desc->SetDivisor(H * W);
    // if (status == musa::dnn::Status::SUCCESS) {
    //     printf("pool_desc SetDivisor status:%d\n", static_cast<int>(status));
    // }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    *desc_ptr = new GlobalAvgPoolMusaDescriptor{
        DevMtGpu,
        y->dt,
        handle->device_id,
        ndim,
        0,
        0,
        0,
        0,
        0,
        0,
        handle->mudnn_handles_t,
        x_tensor,
        y_tensor,
        pool_desc,
        indices,
        alpha,
        beta,
    };

    delete[] x_shape;
    delete[] y_shape;

    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetGlobalAvgPoolWorkspaceSize(GlobalAvgPoolMusaDescriptor_t desc, uint64_t *size) {
    *size = desc->ndim <= 5 ? 0 : (desc->dtype != F16 ? 0 : std::min(desc->dtype.size * 2, 8) * desc->y_data_size);
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyGlobalAvgPoolDescriptor(GlobalAvgPoolMusaDescriptor_t desc) {
    if (desc->ndim <= 5) {
        delete desc->x_desc;
        delete desc->y_desc;
        delete desc->pool_desc;
        delete desc->indices;
    }
    desc->mudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
