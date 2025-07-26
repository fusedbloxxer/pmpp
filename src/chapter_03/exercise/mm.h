#include <torch/torch.h>

namespace chapter_03::mm
{
    torch::Tensor mm_row(const torch::Tensor &a, const torch::Tensor &b);
    torch::Tensor mm_col(const torch::Tensor &a, const torch::Tensor &b);
    torch::Tensor mv(const torch::Tensor &a, const torch::Tensor &b);
}