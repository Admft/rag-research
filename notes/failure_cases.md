# Failure case notes

Record retrieval and generation failures here. These are research signals, not noise.

## Template

```markdown
### YYYY-MM-DD — short title

- Query:
- Expected:
- Actual:
- Retrieved sources:
- Likely cause:
- Follow-up:
```

---

## Examples to watch for

- Retrieval misses the right document entirely
- Right document retrieved but wrong chunk rank
- Chunk too small — answer split across chunks
- Chunk too large — irrelevant text dilutes the match
- Generator ignores retrieved context
- Generator hallucinates despite good retrieval

---

_Add your failure cases below._
