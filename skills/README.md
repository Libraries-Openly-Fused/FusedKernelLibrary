# FKL Agent Skills

Machine- and human-readable guides for working with the Fused Kernel
Library. Each `SKILL.md` follows the agent-skills convention (YAML
frontmatter with `name` + `description`, then a markdown body), so coding
agents (Claude Code, Codex, Cursor, etc.) can load them as procedural
knowledge, and humans can read them as focused docs.

| skill | use it when |
|---|---|
| [`fkl-architecture-overview`](fkl-architecture-overview/SKILL.md) | a 60-second filter: classifying new work as an Operation or DPP before reaching for the deeper skills |
| [`fkl-using-the-library`](fkl-using-the-library/SKILL.md) | writing user code: pipelines, executeOperations, streams, data structures |
| [`fkl-fusion-techniques`](fkl-fusion-techniques/SKILL.md) | choosing/combining Vertical, Backwards Vertical, Horizontal and Divergent Horizontal Fusion |
| [`fkl-implementing-data-parallel-patterns`](fkl-implementing-data-parallel-patterns/SKILL.md) | adding a new DPP struct |
| [`fkl-implementing-operations`](fkl-implementing-operations/SKILL.md) | adding a new Operation struct (Unary/Binary/Read/Write/ReadBack/IncompleteReadBack/Ternary/IncompleteTernary/MidWrite/Open/Closed) |
| [`fkl-data-structures`](fkl-data-structures/SKILL.md) | Ptr2D/Tensor/TensorT/RawPtr, constructors, layouts, external memory |
| [`fkl-build-and-test`](fkl-build-and-test/SKILL.md) | building the repo, running/adding utests, CI expectations |
| [`fkl-language-bindings`](fkl-language-bindings/SKILL.md) | wrapping FKL from another language (Python etc.): JIT strategy, ABI, caching |

Background: the methodology is described in the paper
"The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU
Libraries" (arXiv:2508.07071).
