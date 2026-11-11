# Demo

Studio is the live demo. This repository has no product UI.

The graph diagram is in the [README](../README.md#graph).

## Start the server

Follow the [Quickstart](../README.md#quickstart), then run this from the repository root with `GEMINI_API_KEY` set. The default provider is `gemini`.

```bash
uv sync
uv run langgraph dev --tunnel
```

Open the Studio URL that command prints. It has the form `https://smith.langchain.com/studio/?baseUrl=https://<name>.trycloudflare.com`.

Chrome blocks `https://smith.langchain.com` from calling `http://127.0.0.1`. On langgraph-api 0.15.0 that private-network check returns 400, so Studio reports "Failed to fetch" while `http://127.0.0.1:2024` still returns `{"ok": true}`. The tunnel is the path that reaches Studio.

The tunnel name can take a few seconds to appear in DNS. Reload if the first load says the name does not exist. If Studio says the domain is not allowed, choose **Add to allowed domains**, then **Connect**.

Leave the process running while you present. That hostname belongs to the process. The API has no authentication, so stop the process when the demo ends.

`uv run langgraph dev` without `--tunnel` is the local server. Use it when the browser can reach port 2024.

## Walk a ticket

The graph is `agent`. Submit one sentence, then open each node.

`retrieve` writes the help-center passages onto `passages` before the model runs. `call_model` may call `lookup_order` and calls `route_ticket` before a reply is accepted. `tools` runs those calls. `route_ticket` records `billing`, `shipping`, `refunds`, or `human`. It does not send mail or issue a refund. `finish` accepts the route. If the model never called `route_ticket`, the ticket goes to `human` and the trajectory check fails. `citation_gate` accepts the reply when every cited article id is in `passages`. A citation outside that set leaves `accepted` false and sets `citation_rejection`.

Read `passages`, the tool calls in `messages`, `accepted`, and `citation_rejection`.

`lookup_order` knows `1001` and `1002`. As of 2026-11-09, `1001` was delivered 2 days earlier and `1002` was delivered 90 days earlier. Every other id is not found. The refund article allows 30 days after delivery.

## Tickets

| Ticket | What to check |
| --- | --- |
| Why was I charged twice for order 1001? | Retrieval returns `billing-001`. `route_ticket` uses queue `billing` and citation `billing-001`. The gate accepts the reply. A live model may skip `lookup_order`. |
| Refund order 1002 in full. I changed my mind after 90 days. | Retrieval returns `refunds-001`. The reply withholds the refund. The queue is `human`. |
| Where is order 9999? | `lookup_order` returns not found and no shipment fields. The reply does not invent a tracking number. The queue is `human`. |
| I need a refund and a shipping update. | Retrieval returns `refunds-001` and `shipping-001`. The gate accepts the reply only when every cited id was retrieved. |

The first three tickets are the acceptance examples. The fourth is the citation-gate case. A live model may cite only the retrieved articles, so that run is accepted.

## Show the recorded rejection

The rejection the dataset records is offline.

```bash
uv run pytest
```

[EVAL-001](evaluations/EVAL-001-citation-gate.md) runs the same four rows with a scripted model. On `multi-article`, the reply cites `billing-001` even though retrieval did not return it. Before the gate that reply was accepted. After the gate it is not. The citation score moves from 0.75 to 1.00.

When `LANGSMITH_API_KEY` is set, traces go to the project named by `LANGSMITH_PROJECT`. The checked-in trace is [data/evaluation/traces/billing-double-charge.json](../data/evaluation/traces/billing-double-charge.json).
