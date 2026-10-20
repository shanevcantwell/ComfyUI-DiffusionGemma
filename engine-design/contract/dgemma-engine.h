#ifndef DGEMMA_ENGINE_H
#define DGEMMA_ENGINE_H

/* Proposed facade only. No implementation or stock-llama ABI is supplied. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dgemma_model * dgemma_model_t;
typedef struct dgemma_session * dgemma_session_t;
typedef struct dgemma_cache * dgemma_cache_t;

enum dgemma_status {
    DGEMMA_STATUS_OK          = 0,
    DGEMMA_STATUS_DONE        = 1,
    DGEMMA_STATUS_CANCELLED   = 2,
    DGEMMA_STATUS_INVALID     = -1,
    DGEMMA_STATUS_UNSUPPORTED = -2,
    DGEMMA_STATUS_RESOURCE    = -3,
    DGEMMA_STATUS_STATE       = -4,
    DGEMMA_STATUS_INTERNAL    = -5
};

enum dgemma_stage {
    DGEMMA_STAGE_MODEL      = 0,
    DGEMMA_STAGE_CONSTRAINT = 1,
    DGEMMA_STAGE_SAMPLER    = 2
};

enum dgemma_stop_reason {
    DGEMMA_STOP_REASON_NONE       = 0,
    DGEMMA_STOP_REASON_CONFIDENCE = 1,
    DGEMMA_STOP_REASON_EOS        = 2,
    DGEMMA_STOP_REASON_BUDGET     = 3,
    DGEMMA_STOP_REASON_CANCELLED  = 4,
    DGEMMA_STOP_REASON_ERROR      = 5
};

enum dgemma_embedding_stage {
    DGEMMA_EMBEDDING_STAGE_LOOKUP = 0,
    DGEMMA_EMBEDDING_STAGE_SCALED = 1,
    DGEMMA_EMBEDDING_STAGE_POST_SC = 2
};

enum dgemma_dtype {
    DGEMMA_DTYPE_I32 = 0,
    DGEMMA_DTYPE_F32 = 1,
    DGEMMA_DTYPE_U8  = 2
};

enum dgemma_capability {
    DGEMMA_CAPABILITY_STEP        = 1,
    DGEMMA_CAPABILITY_CACHE       = 2,
    DGEMMA_CAPABILITY_TOPK        = 4,
    DGEMMA_CAPABILITY_FULL_DIST   = 8,
    DGEMMA_CAPABILITY_EMBED_INPUT = 16,
    DGEMMA_CAPABILITY_KV_EXPORT   = 32
};

struct dgemma_tensor_view {
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t rank;
    uint32_t reserved;
    uint64_t shape[4];
    uint64_t stride_bytes[4];
    size_t nbytes;
    const void * data;
};

struct dgemma_model_params {
    uint32_t struct_size;
    uint32_t abi_version;
    const char * gguf_path;
    int32_t gpu_layers;
    uint32_t context_limit;
};

struct dgemma_run_params {
    uint32_t struct_size;
    uint32_t num_steps;
    uint32_t requested_tokens;
    uint32_t stability_steps;
    uint64_t seed;
    float t_min;
    float t_max;
    float entropy_bound;
    float confidence;
    uint32_t capture_top_k;
    uint32_t capture_full_steps;
    uint64_t capture_byte_limit;
};

struct dgemma_step_params {
    uint32_t struct_size;
    uint32_t pin_count;
    float t_min;
    float t_max;
    float entropy_bound;
    uint32_t reserved;
    const int32_t * pin_positions;
    const int32_t * pin_tokens;
};

struct dgemma_frame {
    uint32_t struct_size;
    uint32_t canvas_index;
    uint32_t step_index;
    uint32_t prediction_stage;
    uint64_t generation;
    float schedule_position;
    float temperature;
    float effective_entropy_bound;
    uint32_t stop_reason;
    struct dgemma_tensor_view canvas_sampled;
    struct dgemma_tensor_view canvas_next;
    struct dgemma_tensor_view accepted_mask;
    struct dgemma_tensor_view pinned_mask;
    struct dgemma_tensor_view entropy;
    struct dgemma_tensor_view top_k_ids;
    struct dgemma_tensor_view top_k_weights;
    struct dgemma_tensor_view distribution;
};

uint32_t dgemma_abi_version(void);
enum dgemma_status dgemma_model_init(const struct dgemma_model_params * params, dgemma_model_t * out);
void dgemma_model_free(dgemma_model_t model);
uint64_t dgemma_model_capabilities(dgemma_model_t model);
enum dgemma_status dgemma_session_init(dgemma_model_t model, dgemma_session_t * out);
void dgemma_session_free(dgemma_session_t session);

/* Advance creates a new snapshot; input is borrowed and unchanged. */
enum dgemma_status dgemma_cache_encode(dgemma_session_t session, dgemma_cache_t prefix,
        const int32_t * tokens, size_t count, dgemma_cache_t * out);
void dgemma_cache_free(dgemma_cache_t cache);
enum dgemma_status dgemma_session_begin(dgemma_session_t session, dgemma_cache_t prefix,
        const struct dgemma_run_params * params);

/* Host inputs are consumed or copied before return, never retained by pointer. */
enum dgemma_status dgemma_session_set_input(dgemma_session_t session,
        const struct dgemma_tensor_view * tokens, const struct dgemma_tensor_view * embeddings,
        enum dgemma_embedding_stage embedding_stage);
enum dgemma_status dgemma_session_forward(dgemma_session_t session);
/* Views are read-only and expire on the next session mutation. */
enum dgemma_status dgemma_session_get_prediction(dgemma_session_t session,
        enum dgemma_stage stage, struct dgemma_tensor_view * out);
enum dgemma_status dgemma_session_sample(dgemma_session_t session,
        const struct dgemma_step_params * params, struct dgemma_frame * out);
enum dgemma_status dgemma_session_step(dgemma_session_t session,
        const struct dgemma_step_params * params, struct dgemma_frame * out);
enum dgemma_status dgemma_session_commit(dgemma_session_t session, dgemma_cache_t * out);

/* Only cancellation is concurrently callable on a busy session. */
void dgemma_session_cancel(dgemma_session_t session);
size_t dgemma_session_error(dgemma_session_t session, char * out, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
