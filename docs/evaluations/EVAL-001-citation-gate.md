# EVAL-001: Citation gate

The before citation score is 0.75. One of the four scripted rows is accepted with a citation that retrieval did not return. This note has no after column. The citation gate is not in the graph. The live judge was not run.

The citation score is the share of rows that did not accept an ungrounded citation. Queue accuracy and trajectory accuracy on this run are both 1.00. One ungrounded reply was accepted.

## Before

| Row | Queue | Citation subset | Trajectory | Accepted |
| --- | --- | --- | --- | --- |
| billing-double-charge | pass | pass | pass | yes |
| late-refund | pass | pass | pass | yes |
| missing-order-9999 | pass | pass | pass | yes |
| multi-article | pass | fail | pass | yes |

Rows live in `data/evaluation/dataset.json`. Each row names the expected queue, the expected citation ids, and whether `lookup_order` is required. The runs use the scripted model in that file. No provider is called.

## Known failure

`multi-article` asks for a refund and a shipping update, so retrieval returns `refunds-001` and `shipping-001`. The scripted route cites `refunds-001` and `billing-001`. The subset check fails, and the reply is still accepted.

The late-refund row passes: order 1002 was delivered 90 days before the as-of date, the policy window is 30 days, the queue is `human`, and the reply says the refund is not approved. The same check fails when the queue is `refunds` and the reply approves the refund.

## Why these checks exist

The dataset freezes the queue, the citation ids, and the tool trajectory before the gate changes the graph. The human labels are two replies for each of the four scenarios, one grounded and one not. Agreement with the checked-in fixture judge is 1.00 across those 8 labels. A live judge needs `ANTHROPIC_API_KEY` and `SUPPORT_TRIAGE_LIVE_JUDGE=1`. Without both, the live judge is skipped and this deterministic score still runs.

## Production evaluation

These rows are a regression set for the routing decision and the citation constraint. They are not a sample of production tickets. A production evaluation would need fresh tickets, a human review sample, and a judge run with a model key. This repo keeps the deterministic citation score as the record of the gate.
