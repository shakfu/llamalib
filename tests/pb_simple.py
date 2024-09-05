import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'gemma-2-9b-it-IQ4_XS.gguf'

import pbllama as pb



params = pb.gpt_params()
params.prompt = "Hello my name is"
params.n_predict = 32

# total length of the sequence including the prompt
n_predict: int = params.n_predict

# init LLM

pb.llama_backend_init()
pb.llama_numa_init(params.numa)

# initialize the model

model_params = pb.llama_model_params_from_gpt_params(params)

# set local test model
params.model = str(MODEL)

model = pb.llama_load_model_from_file(params.model, model_params)

if not model:
    raise SystemExit(f"Unable to load model: {params.model}")

# initialize the context

ctx_params = pb.llama_context_params_from_gpt_params(params)

ctx = pb.llama_new_context_with_model(model, ctx_params)

if not ctx:
    raise SystemExit("Failed to create the llama context")

# tokenize the prompt

tokens_list: list[int] = pb.llama_tokenize(ctx, params.prompt, True)

n_ctx: int = pb.llama_n_ctx(ctx)

n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

print("n_predict = %d, n_ctx = %d, n_kv_req = %d", n_predict, n_ctx, n_kv_req)

if (n_kv_req > n_ctx):
    raise SystemExit(
        "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
        "either reduce n_predict or increase n_ctx.")

# print the prompt token-by-token

for i in tokens_list:
    print(pb.llama_token_to_piece(ctx, i))

# create a llama_batch with size 512
# we use this object to submit token data for decoding

batch = pb.llama_batch_init(512, 0, 1)

# evaluate the initial prompt

for i, token in enumerate(tokens_list):
    pb.llama_batch_add(batch, token, i, [], False)


# llama_decode will output logits only for the last token of the prompt
batch.logits[batch.n_tokens - 1] = True

if pb.llama_decode(ctx, batch) != 0:
    raise SystemExit("llama_decode() failed.")

# main loop

n_cur: int    = batch.n_tokens
n_decode: int = 0

t_main_start: int = pb.ggml_time_us()

# ----------------------------------------------------------------------------

n_vocab: int = pb.llama_n_vocab(model)

# logits = pb.llama_get_logits(ctx)

# logits = pb.llama_get_logits_ith(ctx, batch.n_tokens - 1)

# while (n_cur <= n_predict):
#     # sample the next token
#     n_vocab: int = pb.llama_n_vocab(model)
#     # logits: list[float] = pb.llama_get_logits_ith(ctx, batch.n_tokens - 1)
#     logits = pb.llama_get_logits_ith(ctx, batch.n_tokens - 1)

#     candidates: list[llama_token_data] = []
#     # candidates.reserve(n_vocab)

#     for i in range(n_vocab):
#         c = pb.llama_token_data(i, logits[i], 0.0)
#         candidates.append(c)

#     # candidates_p: llama_token_data_array = pb.llama_token_data_array(candidates.data(), len(candidates), False)

#     # # sample the most likely token
#     # new_token_id: llama_token = pb.llama_sample_token_greedy(ctx, &candidates_p)

#     # if (pb.llama_token_is_eog(model, new_token_id) || n_cur == n_predict):
#     #     break

#     # print(pb.llama_token_to_piece(ctx, new_token_id))

#     # # prepare the next batch
#     # pb.llama_batch_clear(batch)

#     # # push this new token for next evaluation
#     # pb.llama_batch_add(batch, new_token_id, n_cur, [], False)

#     n_decode += 1

#     n_cur += 1

#     # evaluate the current batch with the transformer model

#     if (pb.llama_decode(ctx, batch)):
#         raise SystemExit("failed to eval, return code.")

# t_main_end: int = pb.ggml_time_us()

# print("decoded %d tokens in %.2f s, speed: %.2f t/s",
#         n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0))

# pb.llama_print_timings(ctx)

# pb.llama_batch_free(batch)

# pb.llama_free(ctx)
# pb.llama_free_model(model)

# pb.llama_backend_free()
