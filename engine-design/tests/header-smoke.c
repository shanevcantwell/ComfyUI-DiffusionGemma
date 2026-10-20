#include "../contract/dgemma-engine.h"

#ifdef __cplusplus
static_assert(sizeof(int32_t) == 4, "int32 width");
static_assert(sizeof(uint64_t) == 8, "uint64 width");
#else
_Static_assert(sizeof(int32_t) == 4, "int32 width");
_Static_assert(sizeof(uint64_t) == 8, "uint64 width");
#endif

int main(void) {
#ifdef __cplusplus
    dgemma_run_params params = {};
#else
    struct dgemma_run_params params = { 0 };
#endif
    params.struct_size = (uint32_t) sizeof(params);
    return params.struct_size == 0;
}
