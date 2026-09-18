# LLM-Search-Engine

Natural-language to structured-query translation service.

A question in plain English is translated into a constrained JSON query
payload, which the calling application validates and compiles itself. The
translator holds no database connection and no schema of its own — its field
vocabulary is fetched at runtime from the calling application.

## How it works

Translation runs in two stages against a *registry* — a document the calling
application serves, describing the entities and fields a question may refer
to.

1. **Route.** Pick the entity the question is about and the sections of it
   the question touches. This narrows what the second stage is allowed to
   name at all.
2. **Translate.** Hand the model a JSON Schema generated from that narrowed
   slice. Every slot that names a field, a table or an entity is a closed
   enumeration, so a structured-output model cannot name something that does
   not exist. The returned payload is then re-checked against the registry
   for the rules a schema cannot express — which operators suit which field,
   whether an aggregate makes sense over that type, whether a referenced
   table was actually joined.

One retry is spent feeding a validation failure back to the model. After that
the service refuses, with a reason, rather than guessing.

Three answers are possible: the payload; a clarifying question with concrete
candidates, when a value in the question does not resolve to anything the
data holds; or a refusal naming why.

## Endpoints

| Method | Path         | What it does                                       |
| ------ | ------------ | -------------------------------------------------- |
| `GET`  | `/healthz`   | Readiness. Fetches the registry if it has not yet, so a cold process can warm itself; `503` until that succeeds. |
| `POST` | `/translate` | A question in, one of the three answers out. Requires a bearer token. |

`POST /translate` takes `{"question": "...", "now_year": 2026}` — `now_year`
optional, and worth sending: it is what "this year" resolves against.

Authentication is a shared bearer token in the `Authorization` header. It is
required: an instance with no token configured refuses every request rather
than serving them unauthenticated.

## Running it

```sh
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # .venv/Scripts/pip on Windows
cp .env.example .env                   # then fill it in
.venv/bin/uvicorn cert_nlq.api.app:make_app --factory
```

A factory rather than a module-level app, deliberately: importing the module
must not construct a billable client.

## Tests

```sh
.venv/bin/python -m pytest             # .venv/Scripts/python on Windows
.venv/bin/python -m ruff check .
```

The whole suite runs offline and costs nothing. No test makes an API call,
reads an API key, or reaches the network: the model providers are exercised
through a fake, and the registry through a stubbed HTTP transport.

## Configuration

All credentials come from the environment. Copy `.env.example` to `.env` and
fill it in; `.env` is not tracked. No API key belongs in a `.py` file.
