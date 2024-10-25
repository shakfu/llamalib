"""cyllama: a thin cython wrapper of llama.cpp.
"""
import os
import sys
import datetime
from pathlib import Path
import logging

from typing import Optional, Union

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'build'))

import cyllama as cy


Pathlike = Union[str, Path]


# ----------------------------------------------------------------------------
# env helpers

def getenv(key: str, default: bool = False) -> bool:
    """convert '0','1' env values to bool {True, False}"""
    return bool(int(os.getenv(key, default)))

# ----------------------------------------------------------------------------
# constants

PY_VER_MINOR = sys.version_info.minor
DEBUG = getenv('DEBUG', default=True)
COLOR = getenv('COLOR', default=True)

# ----------------------------------------------------------------------------
# logging config

class CustomFormatter(logging.Formatter):
    """custom logging formatting class"""

    white = "\x1b[97;20m"
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(delta)s - %(levelname)s - %(name)s.%(funcName)s - %(message)s"
    cfmt = (f"{white}%(delta)s{reset} - "
            f"{{}}%(levelname)s{{}} - "
            f"{white}%(name)s.%(funcName)s{reset} - "
            f"{grey}%(message)s{reset}")

    FORMATS = {
        logging.DEBUG: cfmt.format(grey, reset),
        logging.INFO: cfmt.format(green, reset),
        logging.WARNING: cfmt.format(yellow, reset),
        logging.ERROR: cfmt.format(red, reset),
        logging.CRITICAL: cfmt.format(bold_red, reset),
    }

    def __init__(self, use_color=COLOR):
        self.use_color = use_color

    def format(self, record):
        """custom logger formatting method"""
        if not self.use_color:
            log_fmt = self.fmt
        else:
            log_fmt = self.FORMATS.get(record.levelno)
        if PY_VER_MINOR > 10:
            duration = datetime.datetime.fromtimestamp(
                record.relativeCreated / 1000, datetime.UTC
            )
        else:
            duration = datetime.datetime.utcfromtimestamp(
                record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S")
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


strm_handler = logging.StreamHandler()
strm_handler.setFormatter(CustomFormatter())
# file_handler = logging.FileHandler("log.txt", mode='w')
# file_handler.setFormatter(CustomFormatter(use_color=False))
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    handlers=[strm_handler],
    # handlers=[strm_handler, file_handler],
)

# ----------------------------------------------------------------------------
# main class

class Llama:
    """top-level api class for llamalib"""

    def __init__(self, model_path: Pathlike, n_predict: int = 512, n_ctx: int = 2048, disable_log: bool = True, n_threads: int = 4):
        self.model_path = Path(model_path)
        self.disable_log = disable_log
        self.log = logging.getLogger(self.__class__.__name__)
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

        # run configuration checks
        self.check_params()
        
        # init LLM
        cy.llama_backend_init()
        cy.llama_numa_init(self.params.numa)

    def __del__(self):
        cy.llama_backend_free()

    def check_params(self):
        if self.params.logits_all:
            self.fail("please use the 'perplexity' tool for perplexity calculations")

        if self.params.embedding:
            self.fail("please use the 'embedding' tool for embedding calculations")

        if self.params.n_ctx != 0 and self.params.n_ctx < 8:
            self.log.warn("minimum context size is 8, using minimum size of 8.")
            self.params.n_ctx = 8;

        if self.params.rope_freq_base != 0.0:
            self.log.warn("changing RoPE frequency base to %g", self.params.rope_freq_base)

        if self.params.rope_freq_scale != 0.0:
            self.log.warn("changing RoPE frequency base to %g", self.params.rope_freq_scale)

    def fail(self, msg, *args):
        """exits the program with an error msg."""
        self.log.critical(msg, *args)
        sys.exit(1)

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
            self.log.info("n_predict = %d, n_ctx = %d, n_kv_req = %d" % (n_predict, n_ctx, n_kv_req))

        if (n_kv_req > n_ctx):
            self.fail(
                "error: n_kv_req > n_ctx, the required KV cache size is not big enough\n"
                "either reduce n_predict or increase n_ctx.")

        if not self.disable_log: 
            # print the prompt token-by-token
            prompt=""
            for i in tokens_list:
                prompt += cy.common_token_to_piece(self.ctx, i)
            self.log.info(prompt.strip())

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
            t_main_end: int = cy.ggml_time_us()
            self.log.info("decoded %d tokens in %.2f s, speed: %.2f t/s" %
                    (n_decode, (t_main_end - t_main_start) / 1000000.0, n_decode / ((t_main_end - t_main_start) / 1000000.0)))

        return result.strip()

if __name__ == '__main__':
    MODEL = ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf'
    llm = Llama(model_path=MODEL, disable_log=True)
    prompt = "When did the universe begin?"
    print(prompt)
    print(llm.ask(prompt, n_predict=32))


