#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define RAXIS_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RAXIS_RESTRICT __restrict
#else
#define RAXIS_RESTRICT
#endif