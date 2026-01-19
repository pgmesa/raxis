
#include <raxis/dtype.hpp>


namespace raxis {

    namespace {
        const Dtype* g_default_dtype = &Dtype::Float32;
    }

    const Dtype& default_dtype() {
        return *g_default_dtype;
    }

    void set_default_dtype(const Dtype& dtype) {
        g_default_dtype = &dtype;
    }
    
}