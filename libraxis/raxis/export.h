#pragma once

// A static libraxis (RAXIS_BUILD_STATIC) is part of the program that links
// it: nothing to export or import
#if defined(RAXIS_STATIC)

  #define RAXIS_API

#elif defined(_WIN32) || defined(_WIN64)

  #ifdef RAXIS_BUILD
    #define RAXIS_API __declspec(dllexport)
  #else
    #define RAXIS_API __declspec(dllimport)
  #endif

#else

  #define RAXIS_API __attribute__((visibility("default")))

#endif

// Header-only classes thrown by libraxis and caught in another library (the
// bindings): their typeinfo must stay visible, or each library keeps a hidden
// copy and macOS's libc++ fails the catch (it compares typeinfo addresses).
// MSVC compares type names, and dllexport on a class derived from a std::
// type only adds warning C4275
#if defined(_WIN32) || defined(_WIN64)
  #define RAXIS_VISIBLE
#else
  #define RAXIS_VISIBLE __attribute__((visibility("default")))
#endif