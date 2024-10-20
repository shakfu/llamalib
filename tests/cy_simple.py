import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'

import cyllama as cy

params = cy.CommonParams()
params.model = str(MODEL)
params.prompt = "When did the universe begin?"
params.n_predict = 32
params.n_ctx = 2048
params.cpuparams.n_threads = 4



# args = []
# if not cy.gpt_params_parse(args, params, cy.LLAMA_EXAMPLE_COMMON):
#     raise SystemExit("gpt_params_parse failed")

# total length of the sequence including the prompt
n_predict: int = params.n_predict

# init LLM

cy.llama_backend_init()
cy.llama_numa_init(params.numa)

# initialize the model

model_params = cy.common_model_params_to_llama(params)

# set local test model
params.model = str(MODEL)

# model = cy.llama_load_model_from_file(params.model, model_params)
model = cy.LlamaModel(path_model=params.model, params=model_params)

if not model:
    raise SystemExit(f"Unable to load model: {params.model}")

# initialize the context

ctx_params = cy.common_context_params_to_llama(params)

# ctx = cy.llama_new_context_with_model(model, ctx_params)
ctx = cy.LlamaContext(model=model, params=ctx_params)

if not ctx:
    raise SystemExit("Failed to create the llama context")


sparams = cy.llama_sampler_chain_default_params()

sparams.no_perf = False

# smplr = cy.llama_sampler_chain_init(sparams)
smplr = cy.LlamaSampler(sparams)

if not smplr:
    raise SystemExit(f"Unable to init sampler.")


# cy.llama_sampler_chain_add(smplr, cy.llama_sampler_init_greedy())
smplr.add_greedy()


# tokenize the prompt

tokens_list: list[int] = cy.common_tokenize(ctx, params.prompt, True)

n_ctx: int = ctx.n_ctx()

n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

print("n_predict = %d, n_ctx = %d, n_kv_req = %d" % (n_predict, n_ctx, n_kv_req))

if (n_kv_req > n_ctx):
    raise SystemExit(
        "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
        "either reduce n_predict or increase n_ctx.")

# print the prompt token-by-token
print()
prompt=""
for i in tokens_list:
    prompt += cy.common_token_to_piece(ctx, i)
print(prompt)

# create a llama_batch with size 512
# we use this object to submit token data for decoding

# batch = cy.llama_batch_init(512, 0, 1)
batch = cy.LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)

# evaluate the initial prompt
for i, token in enumerate(tokens_list):
    cy.common_batch_add(batch, token, i, [0], False)

# llama_decode will output logits only for the last token of the prompt
# batch.logits[batch.n_tokens - 1] = True
batch.set_last_logits_to_true()

# logits = batch.get_logits()

if cy.llama_decode(ctx, batch) != 0:
    raise SystemExit("llama_decode() failed.")

# main loop

n_cur: int    = batch.n_tokens
n_decode: int = 0

t_main_start: int = cy.ggml_time_us()

result: str = ""

while (n_cur <= n_predict):
    # sample the next token

    if True:
        new_token_id = smplr.sample(ctx, batch.n_tokens - 1)
        # new_token_id = cy.llama_sampler_sample(smplr, ctx, batch.n_tokens - 1)

        # print("new_token_id: ", new_token_id)

        # cy.llama_sampler_accept(smplr, new_token_id);
        smplr.accept(new_token_id)

        # is it an end of generation?
        if (cy.llama_token_is_eog(model, new_token_id) or n_cur == n_predict):
            print()
            break

        result += cy.common_token_to_piece(ctx, new_token_id)

        # prepare the next batch
        cy.common_batch_clear(batch);

        # push this new token for next evaluation
        cy.common_batch_add(batch, new_token_id, n_cur, [0], True)

        n_decode += 1

    n_cur += 1

    # evaluate the current batch with the transformer model

    if cy.llama_decode(ctx, batch):
        raise SystemExit("llama_decode() failed.")

print(result)

print()

t_main_end: int = cy.ggml_time_us()

print("decoded %d tokens in %.2f s, speed: %.2f t/s" %
        (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0)))
print()

# cy.llama_perf_sampler_print(smplr)
# cy.llama_perf_context_print(ctx)

# auto cleanup will take care of these
# cy.llama_batch_free(batch)
# cy.llama_sampler_free(smplr)
# cy.llama_free(ctx)
# cy.llama_free_model(model)

cy.llama_backend_free()
