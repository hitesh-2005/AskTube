# AskTube — Task Checklist

## Foundation
- [ ] Keep existing `main.py` unchanged initially.
- [ ] Keep existing `test_main.py`.
- [ ] Keep existing `evaluate_rag.py`.
- [ ] Verify environment setup.
- [ ] Run baseline tests.
- [ ] Run baseline evaluation.

## Application Layer
- [ ] Define video-processing state.
- [ ] Validate URL and request fields.
- [ ] Wrap transcript/index creation.
- [ ] Return structured processing results.
- [ ] Implement question endpoint.
- [ ] Return structured answers and retrieved sections.
- [ ] Map known errors to stable codes.
- [ ] Enforce question length.
- [ ] Add application/API tests.

## Frontend
- [ ] YouTube URL input.
- [ ] Processing states.
- [ ] Video player/context.
- [ ] Ready state.
- [ ] Question input.
- [ ] User messages.
- [ ] AI answers.
- [ ] No-context state.
- [ ] Retrieved transcript accordion.
- [ ] Timestamp pills and seeking.
- [ ] New/change video.
- [ ] Responsive layout.
- [ ] Accessibility.

## Evaluation
- [ ] Record baseline evaluation.
- [ ] Evaluate chunking alternatives.
- [ ] Evaluate k.
- [ ] Evaluate threshold.
- [ ] Test multilingual retrieval.
- [ ] Keep only evidence-backed changes.

## Quality
- [ ] Preserve prompt injection defense.
- [ ] Preserve deterministic no-context behavior.
- [ ] Preserve cache validation.
- [ ] Do not expose secrets.
- [ ] Keep dependencies minimal.
- [ ] Update docs when contracts change.

## Final
- [ ] Clean setup.
- [ ] Tests pass.
- [ ] Evaluation runs.
- [ ] Frontend/backend integration works.
- [ ] README complete.
- [ ] No unnecessary infrastructure added.
