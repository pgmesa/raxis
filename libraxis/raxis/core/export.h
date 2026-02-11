#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #define RAXIS_API __declspec(dllexport)
#else
    #define RAXIS_API __attribute__((visibility("default")))
#endif