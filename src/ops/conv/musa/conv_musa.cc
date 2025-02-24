#include "conv_musa.h"
#include "../../../devices/musa/common_musa.h"
#include "../../utils.h"

infiniopStatus_t musaCreateConvDescriptor(MusaHandle_t handle,
                                          ConvMusaDescriptor_t *desc_ptr,
                                          infiniopTensorDescriptor_t y,
                                          infiniopTensorDescriptor_t x,
                                          infiniopTensorDescriptor_t w,
                                          void const *pads,
                                          void const *strides,
                                          void const *dilations,
                                          uint64_t n) {
    uint64_t ndim = y->ndim;
    if (ndim < 3 || ndim != x->ndim || ndim != w->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (ndim > 5 ) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (x->shape[0] != y->shape[0] || w->shape[0] != y->shape[1] || x->shape[1] != w->shape[1]) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt || y->dt != w->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    const uint64_t new_ndim = std::max(ndim, (uint64_t)4);
    // convert pads, strides, dilations into int32[]
    int32_t *pad = new int32_t[new_ndim];
    int32_t *stride = new int32_t[new_ndim];
    int32_t *dilation = new int32_t[new_ndim];
    int64_t *x_shape = new int64_t[new_ndim];
    int64_t *w_shape = new int64_t[new_ndim];
    int64_t *y_shape = new int64_t[new_ndim];
    auto pads_ = reinterpret_cast<uint64_t const *>(pads);
    auto strides_ = reinterpret_cast<int64_t const *>(strides);
    auto dilations_ = reinterpret_cast<uint64_t const *>(dilations);
    for (size_t i = 0; i < new_ndim; ++i) {
        pad[i] = i < ndim - 2 ? static_cast<int>(pads_[i]) : 0;
        stride[i] = i < ndim - 2 ? static_cast<int>(strides_[i]) : 1;
        dilation[i] = i < ndim - 2 ? static_cast<int>(dilations_[i]) : 1;
        x_shape[i] = i < ndim ? static_cast<int64_t>(x->shape[i]) : 1;
        w_shape[i] = i < ndim ? static_cast<int64_t>(w->shape[i]) : 1;
        y_shape[i] = i < ndim ? static_cast<int64_t>(y->shape[i]) : 1;
    }

    musa::dnn::Tensor *x_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *y_tensor = new musa::dnn::Tensor();
    musa::dnn::Tensor *w_tensor = new musa::dnn::Tensor();

    if (y->dt == F16) {
        x_tensor->SetType(musa::dnn::Tensor::Type::HALF);
        y_tensor->SetType(musa::dnn::Tensor::Type::HALF);
        w_tensor->SetType(musa::dnn::Tensor::Type::HALF);
    } else if (y->dt == F32) {
        x_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
        y_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
        w_tensor->SetType(musa::dnn::Tensor::Type::FLOAT);
    }

    if (new_ndim == 5) {
        x_tensor->SetFormat(musa::dnn::Tensor::Format::NCDHW);
        y_tensor->SetFormat(musa::dnn::Tensor::Format::NCDHW);
    }
    else if (new_ndim == 4) {
        x_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
        y_tensor->SetFormat(musa::dnn::Tensor::Format::NCHW);
    }
    else if (new_ndim == 3) {
        x_tensor->SetFormat(musa::dnn::Tensor::Format::NCW);
        y_tensor->SetFormat(musa::dnn::Tensor::Format::NCW);
    }
    else {
        return STATUS_BAD_TENSOR_SHAPE;
    }

    x_tensor->SetNdInfo((int) new_ndim, x_shape);
    y_tensor->SetNdInfo((int) new_ndim, y_shape);
    w_tensor->SetNdInfo((int) new_ndim, w_shape);

    // musa::dnn::Status status1 = y_tensor->SetNdInfo((int) new_ndim, y_shape);
    // if (status1 == musa::dnn::Status::SUCCESS) {
    //     std::cerr << "Success to set y_tensor." << std::endl;
    // }

    // 设置卷积的填充、步长和膨胀
    musa::dnn::Convolution* conv_operator = new musa::dnn::Convolution();
    musa::dnn::Status status2 = conv_operator->SetNdInfo(new_ndim - 2, pad, stride, dilation);
    // if (status2 == musa::dnn::Status::SUCCESS) {
    //     std::cerr << "Success to set convolution dimensions." << std::endl;
    // }

    musa::dnn::Status status3 = conv_operator->SetComputeMode(musa::dnn::Convolution::ComputeMode::TENSOR);
    // if (status3 == musa::dnn::Status::SUCCESS) {
    //     std::cerr << "Success to set compute mode." << std::endl;
    //     // printf("status3: %s\n",status3);
    //     printf("SetComputeMode Status:%d\n", static_cast<int>(status3));
    // }    

    musa::dnn::Convolution::Algorithm algo = musa::dnn::Convolution::Algorithm::DIRECT;


    use_mudnn(handle->mudnn_handles_t, handle->device_id, nullptr, 
                [&](musa::dnn::Handle* handle) {conv_operator->GetRecommendForwardAlgorithm(*handle, algo, *y_tensor, *x_tensor, *w_tensor);});

    size_t workspace_size = 2;
    // printf("workspace_size before: %zu\n", workspace_size);

    use_mudnn(handle->mudnn_handles_t, handle->device_id, nullptr, 
                [&](musa::dnn::Handle* handle) {
                musa::dnn::Status status = conv_operator->GetForwardWorkspaceSize(*handle, workspace_size, *y_tensor, *x_tensor, *w_tensor, algo);
                // printf("GetForwardWorkspaceSize status: %d\n", static_cast<int>(status));
            });

    // printf("workspace_size after: %zu\n", workspace_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    musa::dnn::MemoryMaintainer maintainer = [](size_t size) -> musa::dnn::MemoryHandler {
        void* ptr = nullptr;
        musaMalloc(&ptr, size);  
        return musa::dnn::MemoryHandler(ptr, [](void* p) {
            if (p) musaFree(p); 
        });
    };

    // musa::dnn::MemoryHandler workspace_mem  = maintainer(workspace_size);

    *desc_ptr = new ConvMusaDescriptor{
        DevMtGpu,
        y->dt,
        handle->device_id,
        handle->mudnn_handles_t,
        x_tensor,
        w_tensor,
        y_tensor,
        conv_operator,
        algo,
        alpha,
        beta,
        workspace_size,
        maintainer
        };

    delete[] pad;
    delete[] stride;
    delete[] dilation;
    delete[] x_shape;
    delete[] w_shape;
    delete[] y_shape;

    return STATUS_SUCCESS;
}

infiniopStatus_t musaGetConvWorkspaceSize(ConvMusaDescriptor_t desc, uint64_t *size) {
    *size = desc->workspace_size;
    return STATUS_SUCCESS;
}

infiniopStatus_t musaDestroyConvDescriptor(ConvMusaDescriptor_t desc) {

    desc->mudnn_handles_t = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
