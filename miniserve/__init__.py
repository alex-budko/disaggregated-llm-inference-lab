"""miniserve: a tiny educational LLM serving stack.

Three concepts demonstrated here:
  1. KV cache speeds up decoding (avoid re-attending over the full prompt
     for every new token).
  2. Prefix caching skips prefill on repeated long prompts.
  3. Disaggregated prefill runs the prompt forward pass and the
     token-by-token generation on separate workers so they don't
     head-of-line-block each other under concurrent load.
"""

__version__ = "0.1.0"
