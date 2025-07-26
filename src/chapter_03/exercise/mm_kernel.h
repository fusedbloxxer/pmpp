#pragma once

namespace chapter_03::mm_kernel
{
    void mm_row(float *a, float *b, float *c, uint N);
    void mm_col(float *a, float *b, float *c, uint N);
    void mv(float *a, float *b, float *c, uint N);
}
