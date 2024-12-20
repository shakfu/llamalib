import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'build'))

import nbllama as nb


def ask(prompt, model, n_predict=512, disable_log=True, n_threads=4) -> str:
    """ask/prompt a llama model"""
    return nb.simple_prompt(model=model, n_predict=n_predict, prompt=prompt, disable_log=disable_log,  n_threads=n_threads).strip()

def test_nb_highlevel_simple(model_path):
    ask("When did the universe begin?", model=model_path)
    assert True


def test_nb_lowlevel_simple(model_path):

    params = nb.common_params()
    params.model = model_path
    params.prompt = "When did the universe begin?"
    params.n_predict = 32
    params.n_ctx = 2048

    args = []
    if not nb.common_params_parse(args, params, nb.LLAMA_EXAMPLE_COMMON):
        raise SystemExit("common_params_parse failed")

    # total length of the sequence including the prompt
    n_predict: int = params.n_predict

    # init LLM

    nb.llama_backend_init()
    nb.llama_numa_init(params.numa)

    # initialize the model

    model_params = nb.common_model_params_to_llama(params)

    # set local test model
    params.model = model_path

    model = nb.llama_load_model_from_file(params.model, model_params)

    if not model:
        raise SystemExit(f"Unable to load model: {params.model}")

    # initialize the context

    ctx_params = nb.common_context_params_to_llama(params)

    ctx = nb.llama_new_context_with_model(model, ctx_params)

    if not ctx:
        raise SystemExit("Failed to create the llama context")


    sparams = nb.llama_sampler_chain_default_params()

    sparams.no_perf = False

    smpl = nb.llama_sampler_chain_init(sparams)

    if not smpl:
        raise SystemExit(f"Unable to init sampler.")


    nb.llama_sampler_chain_add(smpl, nb.llama_sampler_init_greedy())


    # tokenize the prompt

    tokens_list: list[int] = nb.common_tokenize(ctx, params.prompt, True)

    n_ctx: int = nb.llama_n_ctx(ctx)

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
        prompt += nb.common_token_to_piece(ctx, i)
    print(prompt)

    # create a llama_batch with size 512
    # we use this object to submit token data for decoding

    batch = nb.llama_batch_init(512, 0, 1)

    # evaluate the initial prompt
    for i, token in enumerate(tokens_list):
        nb.common_batch_add(batch, token, i, [0], False)

    # llama_decode will output logits only for the last token of the prompt
    # logits = batch.get_logits()
    # logits[batch.n_tokens - 1] = True
    # batch.logits[batch.n_tokens - 1] = True
    batch.set_last_logits_to_true()

    # logits = batch.get_logits()

    if nb.llama_decode(ctx, batch) != 0:
        raise SystemExit("llama_decode() failed.")

    # main loop

    n_cur: int    = batch.n_tokens
    n_decode: int = 0

    t_main_start: int = nb.ggml_time_us()

    result: str = ""

    while (n_cur <= n_predict):
        # sample the next token
        if True:
            new_token_id = nb.llama_sampler_sample(smpl, ctx, batch.n_tokens - 1)

            # print("new_token_id: ", new_token_id)

            nb.llama_sampler_accept(smpl, new_token_id);

            # is it an end of generation?
            if (nb.llama_token_is_eog(model, new_token_id) or n_cur == n_predict):
                print()
                break

            result += nb.common_token_to_piece(ctx, new_token_id)

            # prepare the next batch
            nb.common_batch_clear(batch);

            # push this new token for next evaluation
            nb.common_batch_add(batch, new_token_id, n_cur, [0], True)

            n_decode += 1

        n_cur += 1

        # evaluate the current batch with the transformer model
        if nb.llama_decode(ctx, batch):
            raise SystemExit("llama_decode() failed.")

    print(result)

    print()

    t_main_end: int = nb.ggml_time_us()

    print("decoded %d tokens in %.2f s, speed: %.2f t/s",
            n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0))
    print()


    nb.llama_batch_free(batch)
    nb.llama_sampler_free(smpl)
    nb.llama_free(ctx)
    nb.llama_free_model(model)

    nb.llama_backend_free()

    assert True

