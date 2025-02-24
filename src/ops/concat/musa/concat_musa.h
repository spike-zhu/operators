#ifndef __MUSA_CONCAT_H__
#define __MUSA_CONCAT_H__

#include "../../../devices/musa/common_musa.h"
#include "../../../devices/musa/musa_handle.h"
#include "operators.h"
#include <vector>
#include <numeric>

struct ConcatMusaDescriptor {
    Device device;
    DT dtype;
    int64_t axis;
    uint64_t num_inputs;
    std::vector<std::vector<uint64_t>> input_shapes;  
    std::vector<uint64_t> output_shape;          
};

typedef struct ConcatMusaDescriptor *ConcatMusaDescriptor_t;

infiniopStatus_t musaCreateConcatDescriptor(MusaHandle_t handle,
                                            ConcatMusaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t *x,
                                            uint64_t nums_input,
                                            int64_t axis);

infiniopStatus_t musaConcat(ConcatMusaDescriptor_t desc,
                            void *y,
                            void const **x,
                            void *stream);

infiniopStatus_t musaDestroyConcatDescriptor(ConcatMusaDescriptor_t desc);

#endif