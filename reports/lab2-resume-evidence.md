# Lab 2 — Checkpoint and Resume Evidence

The tuning workflow uses a checkpoint file to record completed trials and cumulative spend.

During the Lab 2 tuning study, the workflow was restarted after an earlier trial had already completed. On restart, the completed trial was detected from the checkpoint and skipped instead of being trained again.

Observed resume behaviour:

```text
trial 0: already done, skipping (resumed from checkpoint)


```

This demonstrates that completed work survives an interruption and that the study can continue from the saved checkpoint rather than restarting from the beginning.
