RAG Research — Results
======================

Everything from every run lives here. One folder per run.

STRUCTURE
---------

experiments/Results/
  MASTER_LOG.txt           Start here. Compares ALL runs.
  summary.csv              Spreadsheet-friendly table of all runs.
  000-005_FULL_REPORT.txt  Baseline progression report (research write-up).
  runs/
    000__og_baseline__2026-06-22_.../
    001__baseline__2026-06-22_.../
    002__baseline__2026-06-22_.../
      REPORT.txt           Human-readable report for THIS run.
      data.json            Full raw data.
      questions.jsonl      Per-question details.

INSIDE EACH RUN FOLDER
----------------------

  REPORT.txt       Read this first.
  data.json        Machine-readable full output.
  questions.jsonl  One line per eval question (experiments + eval runs).

NAMING
------

  {number}__{run_name}__{date}_{time}_CDT

  Examples:
    000__og_baseline__2026-06-22_23-40-42_CDT   (original baseline)
    001__baseline__2026-06-22_20-28-46_CDT      (grid smoke test)
    004__baseline__2026-06-22_22-51-27_CDT      (best grid baseline)

  Run 000 is reserved for the OG baseline (src/run_og_baseline.py).
  Runs 001+ are from the experiment grid (src/run_experiments.py).

RUN TYPES
---------

  og_baseline          Original first-dev baseline (run 000)
  experiment           Grid test (chunk size, top-k, retriever, etc.)
  build_index          Indexed documents into Qdrant
  evaluate_retrieval   Baseline retrieval eval (Recall@k, MRR@k)
  generation           Single question answered via Ollama

SCORES
------

  Retrieval only  → higher Recall@k / MRR@k is better
  Full pipeline   → higher final_score is better (0-100)

  Full pipeline metrics in REPORT.txt:
    final_score, answer_correctness, faithfulness,
    context_recall, context_precision, citation_accuracy,
    answer_parse_rate, Recall@k, MRR@k, avg latency

BASELINE PROGRESSION (000–005)
------------------------------

  000  OG baseline — chunk 120/30, no reranker, og_strict prompt
  001  Grid smoke test — retrieval-only, 2 questions
  002  First full grid baseline — chunk 256, BGE reranker, citation prompt
  003  Scratchpad/answer XML tags — citation accuracy 38% → 94%
  004  Token budget + scratchpad brevity — best final score (74.39)
  005  Re-run of 004 config — confirms ~74 final score

  Details: 000-005_FULL_REPORT.txt

REGENERATE MASTER LOG
---------------------

  python src/regenerate_experiment_log.py
