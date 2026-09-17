# AskTube — RAG Evaluation Plan

## Purpose
Measure retrieval quality before changing chunking, k, or threshold.

## Existing Evaluation
`evaluate_rag.py` is the real retrieval evaluation/calibration script. Keep it separate from `test_main.py`.

- `test_main.py`: fast, network-free logic/unit tests.
- `evaluate_rag.py`: actual embeddings + FAISS + production retriever evaluation.

The current dataset contains hand-checkable Python, TensorFlow, and clustering questions plus no-context questions about the capital of France and tomorrow's weather. Expected timestamp ranges are defined in `evaluate_rag.py`.

## Baseline
- chunk size: 1000
- overlap: 200
- k: 4
- threshold: 0.5

## Experiments
Try nearby alternatives such as:
- 700/150, 800/160, 1200/240
- k 3/4/5
- threshold 0.4/0.5/0.6

Do not change production defaults until results show a meaningful overall improvement.

## Metrics
Current dataset:
- expected-region retrieval success
- no-context rejection
- false-positive retrieval
- qualitative grounding

As the dataset grows:
- Recall@k
- Precision@k
- hit rate
- no-context accuracy

Eventually include direct factual, paraphrased, multilingual, cross-chunk, and answer-absent questions across several videos.

## Commands
```bash
python test_main.py
python evaluate_rag.py
python evaluate_rag.py --k 5 --threshold 0.4
python evaluate_rag.py --chunk-size 700 --chunk-overlap 150
```
