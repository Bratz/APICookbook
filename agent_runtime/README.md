# agent_runtime: decode, APICookbook fork

The agent behind APICookbook's "Build & run" mode: a trimmed copy of
[decode](https://github.com/decodingai-magazine/building-a-coding-agent-from-scratch-course)
(Apache-2.0), running Claude in a Modal sandbox. Upstream commit and every change: [NOTICE](NOTICE)
and [docs/adr/0024-apicookbook-fork.md](docs/adr/0024-apicookbook-fork.md). Plan:
[../docs/decode-fork-plan.md](../docs/decode-fork-plan.md).

```bash
uv sync                      # Python 3.12
uv run pytest tests/unit     # the kept upstream tests
uv run decode run "<task>"   # one headless run (needs ANTHROPIC_API_KEY)
```
