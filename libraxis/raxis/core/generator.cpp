
#include <raxis/core/generator.h>


namespace raxis {

Generator& get_default_generator() {
    static Generator instance(kDefaultSeed);
    return instance;
}

void manual_seed(uint64_t seed) {
    get_default_generator().seed(seed);
}

} // namespace raxis