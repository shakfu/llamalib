"""
cyllama: a thin cython wrapper of llama.cpp

classes:
    LlamaTokenData
    LlamaTokenDataArray
    LoraAdapter
    GGMLTensor
    SamplerChainParams
    LlamaSampler
    CommonSamplerParams
    CommonSampler
    CpuParams
    CommonParams
    ModelParams
    ModelQuantizeParams
    LlamaModel
    ContextParams
    LlamaContext
    LlamaBatch
"""

import sys
from pathlib import Path

from typing import Optional, Union

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

import cyllama as cy


Pathlike = Union[str, Path]


class Llama:
    """top-level api class for llamalib"""

    def __init__(self, model_path: Pathlike, n_predict: int = 512, n_ctx: int = 2048, disable_log: bool = True, n_threads: int = 4):
        self.model_path = Path(model_path)
        self.disable_log = disable_log
        if not self.model_path.exists():
            raise SystemExit(f"Provided model does not exist: {model_path}")
        
        self.params = cy.CommonParams()
        self.params.model = str(self.model_path)
        self.params.n_predict = n_predict
        self.params.n_ctx = n_ctx
        self.params.verbosity = -1;
        self.params.cpuparams.n_threads = n_threads

        self.model: Optional[cy.LlamaModel] = None
        self.ctx: Optional[cy.LlamaContext] = None
        self.smplr: Optional[cy.LlamaSampler] = None
        
        if self.disable_log:
            cy.log_set_verbosity(self.params.verbosity)

        cy.common_init()
        
        # init LLM
        cy.llama_backend_init()
        cy.llama_numa_init(self.params.numa)

    def __del__(self):
        cy.llama_backend_free()

    def ask(self, prompt: str, n_predict: Optional[int] = None, n_ctx: Optional[int] = None):
        """prompt model"""

        self.params.prompt = prompt

        if n_predict:
            self.params.n_predict = n_predict
        if n_ctx:
            self.params.n_ctx = n_ctx

        # total length of the sequence including the prompt
        n_predict: int = self.params.n_predict

        # initialize the model

        model_params = cy.common_model_params_to_llama(self.params)

        self.model = cy.LlamaModel(path_model=self.params.model, params=model_params)

        # initialize the context
        ctx_params = cy.common_context_params_to_llama(self.params)
        self.ctx = cy.LlamaContext(model=self.model, params=ctx_params)

        # build sampler chain
        sparams = cy.SamplerChainParams()
        sparams.no_perf = False

        self.smplr = cy.LlamaSampler(sparams)
        self.smplr.add_greedy()

        # tokenize the prompt
        tokens_list: list[int] = cy.common_tokenize(self.ctx, self.params.prompt, True)
        n_ctx: int = self.ctx.n_ctx()
        n_kv_req: int = len(tokens_list) + (n_predict - len(tokens_list))

        if not self.disable_log:
            print("n_predict = %d, n_ctx = %d, n_kv_req = %d" % (n_predict, n_ctx, n_kv_req))

        if (n_kv_req > n_ctx):
            raise SystemExit(
                "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
                "either reduce n_predict or increase n_ctx.")

        if not self.disable_log: 
            # print the prompt token-by-token
            print()
            prompt=""
            for i in tokens_list:
                prompt += cy.common_token_to_piece(self.ctx, i)
            print(prompt)

        # create a llama_batch with size 512
        # we use this object to submit token data for decoding

        # create batch
        batch = cy.LlamaBatch(n_tokens=512, embd=0, n_seq_max=1)

        # evaluate the initial prompt
        for i, token in enumerate(tokens_list):
            cy.common_batch_add(batch, token, i, [0], False)

        # llama_decode will output logits only for the last token of the prompt
        # batch.logits[batch.n_tokens - 1] = True
        batch.set_last_logits_to_true()

        # logits = batch.get_logits()

        self.ctx.decode(batch)

        # main loop

        n_cur: int    = batch.n_tokens
        n_decode: int = 0
        result: str = ""

        if not self.disable_log:
            t_main_start: int = cy.ggml_time_us()

        while (n_cur <= n_predict):
            # sample the next token

            if True:
                new_token_id = self.smplr.sample(self.ctx, batch.n_tokens - 1)

                self.smplr.accept(new_token_id)

                # is it an end of generation?
                if (self.model.token_is_eog(new_token_id) or n_cur == n_predict):
                    if not self.disable_log:
                        print()
                    break

                result += cy.common_token_to_piece(self.ctx, new_token_id)

                # prepare the next batch
                cy.common_batch_clear(batch);

                # push this new token for next evaluation
                cy.common_batch_add(batch, new_token_id, n_cur, [0], True)

                n_decode += 1

            n_cur += 1

            # evaluate the current batch with the transformer model
            self.ctx.decode(batch)


        if not self.disable_log:
            print()
            t_main_end: int = cy.ggml_time_us()
            print("decoded %d tokens in %.2f s, speed: %.2f t/s" %
                    (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0)))
            print()

        return result.strip()

if __name__ == '__main__':
    MODEL = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'
    llm = Llama(model_path=MODEL)
    prompt = "When did the universe begin?"
    print(prompt)
    print(llm.ask(prompt, n_predict=32))


