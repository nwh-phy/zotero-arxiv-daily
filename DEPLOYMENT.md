# Physics literature digest deployment

The fork runs on GitHub-hosted Ubuntu with Python 3.13 and CPU PyTorch. The original recency-weighted embedding ranking is preserved. Scheduled trigger: `15 0 * * 1` (08:15 Asia/Shanghai). GitHub may delay scheduled jobs; this is a trigger time, not a guaranteed mail arrival time.

## Repository settings

Actions Secrets: `ZOTERO_ID`, `ZOTERO_KEY`, `OPENAI_API_KEY`, `OPENAI_API_BASE`, `SENDER`, `RECEIVER`, `SENDER_PASSWORD`.
Actions Variables: `CUSTOM_CONFIG` (optional non-sensitive YAML override), `SMTP_SERVER`, `SMTP_PORT` (465 default), `DELIVERY_ENABLED`.

`CUSTOM_CONFIG` must contain environment references for credentials, never credential values. The workflow does not print it. All credential values belong in Secrets.

The deployed language is Chinese. Abstract-confirmed facts and facts confirmed by the supplied full-text excerpt must be distinguished; the input is capped to 4,000 tokens and is not a complete-paper review. English titles, authors, arXiv publication date/journal metadata and original/PDF links are rendered separately. Summary length and required Chinese fields are validated; at most one repair request is made using the original evidence. Failed, empty or still noncompliant summaries stop delivery instead of quietly substituting English abstracts. Author affiliations are disabled to avoid a second LLM request per paper.

## Complete SMTP setup

Use the repository Settings / Secrets and variables / Actions pages, or `gh secret set NAME -R nwh-phy/zotero-arxiv-daily` with secure interactive input, for sender address, receiver address and SMTP app password. Set `SMTP_SERVER` and `SMTP_PORT` as variables. Never provide a mailbox login password or put credentials in YAML.

Run the `Test` workflow. Its native tests are offline; its live checks cover Zotero, CPU embedding, arXiv, ranking, LLM, Chinese summary, HTML rendering and SMTP authentication. Missing SMTP is explicitly BLOCKED. By default Test does not send email. The optional `send_test` input sends one explicitly labelled deployment test only after all requested checks and SMTP authentication pass; historical samples are labelled as such. SMTP acceptance does not prove inbox arrival. The optional `summary_only` dispatch input retests the summary stage after a summary-only fix; skipped stages must be assessed against the earlier full-run evidence. Verification artifacts are retained for 14 days. The sample uses a real current RSS paper and its abstract; on empty RSS days, a separate arXiv API query supplies real historical records solely for verification. These are labelled as historical samples and never sent by the production workflow; an explicitly requested deployment test may send one labelled historical sample.

Only after SMTP authentication succeeds, set `DELIVERY_ENABLED=true` and manually run `Send weekly literature brief` to verify actual email delivery. If credentials are missing, preflight lists only their names. Until then, the weekly job is deliberately skipped.

## Operations and limits

Keep Alive commits a timestamp monthly to retain repository activity. Native CI tests exclude the model-download test; the Test workflow explicitly exercises the configured embedding model on Ubuntu. Both uv dependencies and Hugging Face downloads are cached. Jobs have bounded timeouts, and LLM retries are limited.

No new papers means no email. A China-date cache marker is saved after SMTP accepts the message, and delivery runs are serialized. This is best-effort duplicate suppression, not transactional exactly-once delivery: cache eviction, cache-save failure or interrupted SMTP acceptance can allow duplicates. Do not blindly rerun a delivery job when SMTP acceptance is uncertain. SMTP acceptance does not establish recipient inbox arrival.

Zotero access is read-only. Corpus filtering uses existing physics-oriented collection paths. No server, Docker or paid embedding resource is provisioned. The LLM reuses an existing OpenAI-compatible service; its availability and quota remain external dependencies.

Weekly delivery: Monday 08:15 Asia/Shanghai (UTC cron `15 0 * * 1`). The arXiv API retrieves all pages for new submissions in the preceding seven complete UTC days, including matching cross-listed categories. Boundaries are UTC midnight (Beijing 08:00). Old-paper revisions are excluded. Abstracts are ranked with the existing algorithm; full-text extraction is attempted only for the top ten. A delayed API index may omit papers; this does not guarantee exhaustive coverage. Timeout is 180 minutes for the larger weekly CPU workload.

Summary generation uses `global:kimi-k3` through the existing Responses service, with low reasoning effort and a 4096-token output budget. The summary itself remains 150–250 Chinese characters. Embedding and weekly ranking are unchanged.
