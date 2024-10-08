# cyllama


```
llama_cpp:
	_internals
		LlamaModel
		LlamaContext
		LlamaBatch
		LlamaTokenDataArray
		LlamaSamplingParams
		LlamaSamplingContext
		CustomSampler
		LlamaSampler
	_utils
		MetaSingleton
		Singleton
	llama
		Llama
		LlamaState
		LogitsProcessor
		LogitsProcessorList
		StoppingCriteria
		StoppingCriteriaList
		MinTokensLogitsProcessor
	llama_cache
		BaseLlamaCache
			LlamaRAMCache
			LlamaDiskCache
	llama_chat_format
		LlamaChatCompletionHandler
		LlamaChatCompletionHandlerNotFoundException
		LlamaChatCompletionHandlerRegistry
		ChatFormatterResponse
		ChatFormatter
		Jinja2ChatFormatter
	llama_cpp
		...
	llama_grammer
		LlamaGrammar
		SchemaConverter
	llama_speculative
		LlamaDraftModel
		LlamaPromptLookupDecoding
	llama_tokenizer
		BaseLlamaTokenizer
			LlamaTokenizer
			LlamaHFTokenizer
	llama_types
		EmbeddingUsage
		Embedding
		CreateEmbeddingResponse
		CompletionLogprobs
		CompletionChoice
		CompletionUsage
		CreateCompletionResponse
		ChatCompletionResponseFunctionCall
		ChatCompletionResponseMessage
		ChatCompletionFunction
		ChatCompletionResponseChoice
		CreateChatCompletionResponse
		ChatCompletionMessageToolCallChunkFunction
		...
```