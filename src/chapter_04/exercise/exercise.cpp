#include "exercise.h"

int chapter_04::exercise::main()
{
    // 1.a. Considering that a block is partitioned in warps of 32 threads and that a block has 128 threads then there are 4 warps per block.
    // 1.b. Considering that there are 8 blocks and 4 warps per block then there are 32 warps in the grid.
    // 1.c.i. A block is divided like the following: 0-31 32-63 64-95 96-127
    //        Then because 64-95 are not executing line 04 then one warp per block is inactive
    //        Then because there are 32 warps and 8 inactive warps in total we have 24 active warps
    // 1.c.ii  32-63 and 96-127 are divergent => 2 warps per block => 16 warps are divergent
    // 1.c.iii 100%
    // 1.c.iv   50% because some threads in the warp 32-63 are executing line 04 and some are not.
    // 1.c.v.   50% because some threads in the warp 96-127 are executing line 04 and some are not.
    // 1.d.i: 32 warps are active
    // 1.d.ii: 32 warps are divergent
    // 1.d.iii: 50% because half of the threads in the warp execute line07 and the others don't
    // 1.e.i: Because there is a modulo operation on i we get three divergent branches
    //        The iterations would be (i%3) = [0 1 2] => 5- [0 1 2] = [5 4 3] => [(0, 4), (0, 3), (0, 2)]
    //        Then 0, 1, 2 have no divergence, because all threads execute those steps => 3 iterations without divergence
    // 1.e.ii: steps i=3 and i=4 have divergence => 2 iterations with divergence

    // 2. Considering a thread block size is 512 threads, to cover the vector length of 2000 we need 4 blocks.
    //    Therefore 4x512 = 2048 threads will be in the grid.

    // 3. The 2048 threads are partitioned into 64 warps of 32 threads each. The array has length 2000
    //    Therefore the warp with divergence will be the one at that 2000 boundary condition.
    //    The warp 62 will have divergence [32 * 62, 32 * 63 - 1] = [1984, 2015]
    //    The final warp does not have divergence but it is inactive.
    //    The final answer is a single warp will have divergence due to bouncary check on vector length.

    // 4. 100 * (3-2 + 3-2.3 + 3-3 + 3 - 2.8 + 3 - 2.4 + 3 - 1.9 + 3 - 2.6 + 3 - 2.9) / 8 * 3 = 17.08% of the total execution time

    // 5. The correctness of executing a kernel should not depend on any assumption that certain threads will
    //    execute in synchrony with each other without the use of barrier synchronizations
    //    I get that the programmer probably assumes that a warp has size 32 and that because of SIMD
    //    the threads will execute the same instruction at once. But the thing is that a thread
    //    might execute an instruction slower due to latency and therefore barrier synchronization is still needed

    // 6. We want to maximize the number of threads in the SM therefore:
    //    a. - 4 * 128 = 512 threads in total
    //    b. - 4 * 256 = 1024 threads in total
    //    c. - 3 * 512 = 1536 threads in total
    //    d. - 1 * 1024 = 1024 threads in total
    //    The answer is c for which 512 threads per block maximizes the number of threads in the SM

    // 7. 64 blocks per SM, 2048 threads per SM
    //    a. 8 blocks with 128 threads each - Yes, 1024/2048 =  50%
    //    b. 16 blocks with 64 threads each - Yes, 1024/2048 =  50%
    //    c. 32 blocks with 32 threads each - Yes, 1024/2048 =  50%
    //    d. 64 blocks with 32 threads each - Yes, 2048/2048 = 100%
    //    e. 32 blocks with 64 threads each - Yes, 2048/2048 = 100%

    // 8. 2048 threads per SM, 32 blocks per SM, 64K registers per SM
    //    a - 128 threads per block, 30r per thread - 16 blocks, 61440 registers in total - Full occupancy!
    //    b -  32 threads per block, 29r per thread - 32 blocks - 1024 threads per block - Half occupancy because of the block size;
    //    c - 256 threads per block, 34r per thread -  8 blocks - 69632 registers in total - Cannot have full occupancy because of number of registers!
    //      - this could be executed if a block was put on another SM => 1792/2048 threads => 87.5% occupancy

    // 9. The student needs 1024x1024 threads because each thread computes one output value in the matrix.
    //    The kernel uses a grid of 32x32 thread blocks => 1024 blocks. One block has 512 threads.
    //    Therefore the number of threads used by the student is 32x32x512 = 1024x512 which is not enough for 1024x1024!
    //    Moreover an SM uses up to 8 blocks. Therefore for 32x32=1024 blocks one would need 128 SMs! not even A100 has that many!
    return 0;
}
