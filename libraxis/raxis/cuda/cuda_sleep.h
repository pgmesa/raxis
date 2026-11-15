#pragma once

#include <cstdint>


namespace raxis::cuda {

/// Queues a kernel that spins for about `cycles` GPU clock cycles on the
/// current stream. Tests use it to keep a stream busy, so that work on
/// another stream would overtake it if the two were not ordered.
/// From: aten/src/ATen/cuda/Sleep.cu (torch.cuda._sleep)
void sleep(int64_t cycles);

} // namespace raxis::cuda
