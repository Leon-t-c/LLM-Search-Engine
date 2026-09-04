# LLM-Search-Engine

Natural-language to structured-query translation service.

A question in plain English is translated into a constrained JSON query
payload, which the calling application validates and compiles itself. The
translator holds no database connection and no schema of its own — its field
vocabulary is fetched at runtime from the calling application.

**Status:** clean start. Implementation pending.

## Configuration

All credentials come from the environment. Copy `.env.example` to `.env` and
fill it in; `.env` is not tracked. No API key belongs in a `.py` file.
