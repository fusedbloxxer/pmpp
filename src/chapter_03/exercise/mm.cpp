#include <torch/torch.h>

#include "mm.h"
#include "mm_kernel.h"

namespace mmk = chapter_03::mm_kernel;

torch::Tensor chapter_03::mm::mm_row(const torch::Tensor &a, const torch::Tensor &b)
{
    torch::Tensor c = torch::empty({a.size(0), b.size(1)});

    float *a_ptr = a.mutable_data_ptr<float>();
    float *b_ptr = b.mutable_data_ptr<float>();
    float *c_ptr = c.mutable_data_ptr<float>();

    mmk::mm_row(a_ptr, b_ptr, c_ptr, a.size(0));

    return c;
}

torch::Tensor chapter_03::mm::mm_col(const torch::Tensor &a, const torch::Tensor &b)
{
    torch::Tensor c = torch::empty({a.size(0), b.size(1)});

    float *a_ptr = a.mutable_data_ptr<float>();
    float *b_ptr = b.mutable_data_ptr<float>();
    float *c_ptr = c.mutable_data_ptr<float>();

    mmk::mm_col(a_ptr, b_ptr, c_ptr, a.size(1));

    return c;
}

torch::Tensor chapter_03::mm::mv(const torch::Tensor &a, const torch::Tensor &b)
{
    torch::Tensor c = torch::empty({b.size(0)});

    float *a_ptr = a.mutable_data_ptr<float>();
    float *b_ptr = b.mutable_data_ptr<float>();
    float *c_ptr = c.mutable_data_ptr<float>();

    mmk::mv(a_ptr, b_ptr, c_ptr, b.size(0));

    return c;
}
