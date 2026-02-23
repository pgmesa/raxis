#pragma once

#if defined(_WIN32) || defined(_WIN64)

  #ifdef RAXIS_BUILD
    #define RAXIS_API __declspec(dllexport)
  #else
    #define RAXIS_API __declspec(dllimport)
  #endif

#else

  #define RAXIS_API __attribute__((visibility("default")))

#endif