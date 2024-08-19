import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'build'))

MODEL = ROOT / 'models' / 'gemma-2-9b-it-IQ4_XS.gguf'

import nbllama as nb

def test_nb_simple():
    params = nb.gpt_params()
    params.prompt = "Hello my name is"
    params.n_predict = 32

    # total length of the sequence including the prompt
    n_predict = params.n_predict

    # init LLM

    nb.llama_backend_init()
    nb.llama_numa_init(params.numa)

    # initialize the model

    model_params: llama_model_params = nb.llama_model_params_from_gpt_params(params)

    # set local test model
    params.model = str(MODEL)

    model: llama_model = nb.llama_load_model_from_file(params.model, model_params)

    if not model:
        raise SystemExit(f"Unable to load model: {params.model}")

    # initialize the context

    ctx_params: llama_context_params = nb.llama_context_params_from_gpt_params(params)

    ctx: llama_context = nb.llama_new_context_with_model(model, ctx_params)

    if not ctx:
        raise SystemExit("Failed to create the llama context")

    # tokenize the prompt

    tokens_list: list[int] = nb.llama_tokenize(ctx, params.prompt, True)

    n_ctx: int = nb.llama_n_ctx(ctx)

    n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

    print("n_predict = %d, n_ctx = %d, n_kv_req = %d", n_predict, n_ctx, n_kv_req)

    if (n_kv_req > n_ctx):
        raise SystemExit(
            "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
            "either reduce n_predict or increase n_ctx.")

    # print the prompt token-by-token

    for i in tokens_list:
        print(nb.llama_token_to_piece(ctx, i))

    # create a llama_batch with size 512
    # we use this object to submit token data for decoding

    batch: llama_batch = nb.llama_batch_init(512, 0, 1)

    # evaluate the initial prompt

    for i, token in enumerate(tokens_list):
        nb.llama_batch_add(batch, token, i, [], False)


    # llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = True

    if nb.llama_decode(ctx, batch) != 0:
        raise SystemExit("llama_decode() failed.")

    # main loop

    n_cur: int    = batch.n_tokens
    n_decode: int = 0

    t_main_start: int = nb.ggml_time_us()

    # while (n_cur <= n_predict):
    #     # sample the next token
    #     n_vocab = nb.llama_n_vocab(model)
    #     logits  = nb.llama_get_logits_ith(ctx, batch.n_tokens - 1)

    #     candidates: list[llama_token_data] = []
    #     # candidates.reserve(n_vocab)

    #     for i in range(n_vocab):
    #         c = nb.llama_token_data(i, logits[i], 0.0)
    #         candidates.append(c)

    #     candidates_p: llama_token_data_array = nb.llama_token_data_array(candidates.data(), len(candidates), False)

    #     # sample the most likely token
    #     new_token_id: llama_token = nb.llama_sample_token_greedy(ctx, &candidates_p)

    #     if (nb.llama_token_is_eog(model, new_token_id) || n_cur == n_predict):
    #         break

    #     print(nb.llama_token_to_piece(ctx, new_token_id))

    #     # prepare the next batch
    #     nb.llama_batch_clear(batch)



    #     # push this new token for next evaluation
    #     nb.llama_batch_add(batch, new_token_id, n_cur, [], False)

    #     n_decode += 1


    #     n_cur += 1

    #     # evaluate the current batch with the transformer model

    #     if (nb.llama_decode(ctx, batch)):
    #         raise SystemExit("failed to eval, return code.")

    t_main_end: int = nb.ggml_time_us()

    print("decoded %d tokens in %.2f s, speed: %.2f t/s",
            n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0))

    nb.llama_print_timings(ctx)

    nb.llama_batch_free(batch)

    # nb.llama_free(ctx)            # CRASH!
    # nb.llama_free_model(model)    # CRASH!

    nb.llama_backend_free()



    # while (n_cur <= n_predict) {
    #     # sample the next token
    #     {
    #         auto   n_vocab = llama_n_vocab(model)
    #         auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1)

    #         std::vector<llama_token_data> candidates
    #         candidates.reserve(n_vocab)

    #         for (llama_token token_id = 0 token_id < n_vocab token_id++) {
    #             candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f })
    #         }

    #         llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false }

    #         # sample the most likely token
    #         const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p)

    #         # is it an end of generation?
    #         if (llama_token_is_eog(model, new_token_id) || n_cur == n_predict) {
    #             LOG_TEE("\n")

    #             break
    #         }

    #         LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str())
    #         fflush(stdout)

    #         # prepare the next batch
    #         llama_batch_clear(batch)

    #         # push this new token for next evaluation
    #         llama_batch_add(batch, new_token_id, n_cur, { 0 }, true)

    #         n_decode += 1
    #     }

    #     n_cur += 1

    #     # evaluate the current batch with the transformer model
    #     if (llama_decode(ctx, batch)) {
    #         fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1)
    #         return 1
    #     }
    # }

    # LOG_TEE("\n")

    # const auto t_main_end = ggml_time_us()

    # LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
    #         __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f))

    # llama_print_timings(ctx)

    # fprintf(stderr, "\n")

    # llama_batch_free(batch)

    # llama_free(ctx)
    # llama_free_model(model)

    # llama_backend_free()

    assert True