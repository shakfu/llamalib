import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'build'))

import pbllama as pb

def test_pb_simple(MODEL):

    params = pb.gpt_params()
    params.model = MODEL
    params.prompt = "When did the universe begin?"
    params.n_predict = 32
    params.n_ctx = 2048


    args = []
    if not pb.gpt_params_parse(args, params, pb.LLAMA_EXAMPLE_COMMON):
        raise SystemExit("gpt_params_parse failed")

    # total length of the sequence including the prompt
    n_predict: int = params.n_predict

    # init LLM

    pb.llama_backend_init()
    pb.llama_numa_init(params.numa)

    # initialize the model

    model_params = pb.llama_model_params_from_gpt_params(params)

    # set local test model
    params.model = MODEL

    model = pb.llama_load_model_from_file(params.model, model_params)

    if not model:
        raise SystemExit(f"Unable to load model: {params.model}")

    # initialize the context

    ctx_params = pb.llama_context_params_from_gpt_params(params)

    ctx = pb.llama_new_context_with_model(model, ctx_params)

    if not ctx:
        raise SystemExit("Failed to create the llama context")


    sparams = pb.llama_sampler_chain_default_params()

    sparams.no_perf = False

    smpl = pb.llama_sampler_chain_init(sparams)

    if not smpl:
        raise SystemExit(f"Unable to init sampler.")


    pb.llama_sampler_chain_add(smpl, pb.llama_sampler_init_greedy())


    # tokenize the prompt

    tokens_list: list[int] = pb.llama_tokenize(ctx, params.prompt, True)

    n_ctx: int = pb.llama_n_ctx(ctx)

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
        prompt += pb.llama_token_to_piece(ctx, i)
    print(prompt)

    # create a llama_batch with size 512
    # we use this object to submit token data for decoding

    batch = pb.llama_batch_init(512, 0, 1)

    # evaluate the initial prompt
    for i, token in enumerate(tokens_list):
        pb.llama_batch_add(batch, token, i, [0], False)

    # llama_decode will output logits only for the last token of the prompt
    # logits = batch.get_logits()
    # logits[batch.n_tokens - 1] = True
    # batch.logits[batch.n_tokens - 1] = True
    batch.set_last_logits_to_true()

    logits = batch.get_logits()

    if pb.llama_decode(ctx, batch) != 0:
        raise SystemExit("llama_decode() failed.")

    # main loop

    n_cur: int    = batch.n_tokens
    n_decode: int = 0

    t_main_start: int = pb.ggml_time_us()

    result: str = ""

    while (n_cur <= n_predict):
        # sample the next token
        if True:
            new_token_id = pb.llama_sampler_sample(smpl, ctx, batch.n_tokens - 1)

            # print("new_token_id: ", new_token_id)

            pb.llama_sampler_accept(smpl, new_token_id);

            # is it an end of generation?
            if (pb.llama_token_is_eog(model, new_token_id) or n_cur == n_predict):
                print()
                break

            result += pb.llama_token_to_piece(ctx, new_token_id)

            # prepare the next batch
            pb.llama_batch_clear(batch);

            # push this new token for next evaluation
            # pb.llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
            pb.llama_batch_add(batch, new_token_id, n_cur, [0], True)

            n_decode += 1

        n_cur += 1

        # evaluate the current batch with the transformer model
        if pb.llama_decode(ctx, batch):
            raise SystemExit("llama_decode() failed.")

    print(result)

    print()

    t_main_end: int = pb.ggml_time_us()

    print("decoded %d tokens in %.2f s, speed: %.2f t/s",
            n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0))
    print()


    pb.llama_batch_free(batch)
    pb.llama_sampler_free(smpl)
    pb.llama_free(ctx)
    pb.llama_free_model(model)

    pb.llama_backend_free()

    assert True
