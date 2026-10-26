# Corpus Organizer PRD — Cohesion-Gated Taxonomy

As of 2026-10-26. Owner: Gui. Review copy with diagrams: https://claude.ai/artifact/X9wYxRSP7CigJbw8i7FkCY
(the engineering spec that preceded this PRD is `docs/ENGINEERING_SPEC.md`).

## Problem

A corpus owner with tens of thousands of items and no labels has two bad options: label by hand, or pay a model to read every item. Both spend effort that scales with the corpus, and most of that effort goes to material that is repetitive and internally consistent.

- Hand labeling 60,000 contract clauses into 100 types takes weeks and a playbook that does not exist yet. The structure has to be discovered before anyone can label against it.
- Reading every item with a model costs one call per item, every time the question changes. On the measured legal corpus that is 60,000 calls per field extracted, with no signal about which answers to trust.
- Clustering tools (BERTopic, TopicGPT, Clio, Nomic Atlas) produce groups and names but give no confidence per group and no cheap way to correct the structure in the owner's own words.

The gap is a product that discovers the organization first, tells the owner which parts of it can be trusted without reading, and lets a correction re-organize the whole corpus.

## Product promise

Point it at a corpus; get an organization you can correct. The product embeds every item locally, clusters them top-down, has a model name only the clusters, and reports a confidence score per group computed before any item is read. The owner renames a group, adds a concept, or seeds one from an item that fits nowhere, and the whole corpus re-organizes around that correction.

One sentence of scope: an unlabeled corpus of text items goes in; a named, navigable hierarchy with per-group confidence, per-item assignments, and optional extracted fields comes out, under the owner's own vocabulary.

What the owner pays for stays flat as the corpus grows: naming a few dozen groups and adjudicating the flagged ones. What the machine pays for scales with the corpus and costs nothing per item. What the model pays for is spent only where the confidence score says a group cannot be trusted.

## Who it is for

The operator is one person who owns a corpus and answers for its organization, not a data team. They know the domain, not clustering. They want to see the structure, correct it in their own words, and know which parts to trust.

| User | Need | Surface |
| --- | --- | --- |
| Corpus owner | Discover, inspect, rename and deepen the organization | Web UI: Discover, Correct, Find, Extract tabs |
| Reviewer | Adjudicate the flagged groups and the items that fit nowhere | Review queue with sample items and the confidence score |
| Downstream consumer | Read assignments, extracted fields, confidence and lineage | Versioned export, HTTP API |
| Researcher | Reproduce every number, challenge a claim | Experiment scripts and pre-registrations |

Three corpus kinds are in scope because they are the three the product has been measured on: legal clauses (LEDGAR, 60,000 items, 100 types), news posts (20 Newsgroups, 6,000 items, 20 groups), and product listings (Amazon, 2,500 items, 5 categories). Any corpus of short text items with a dominant topic per item fits the same path. Items whose attribute of interest is mentioned in passing (periodicity in a clause) do not: cluster-first extraction scored 33.9% against an 80.1% majority baseline there.

## How it works

| Stage | Who pays | Scales with |
| --- | --- | --- |
| 1 Embed and cluster | machine, $0, local | every item |
| 2 Name the groups | model, one call per group | number of groups |
| 3 Gate by cohesion | machine ranks groups; the model reads only the low-cohesion ones | the uncertain share |
| 4 Correct | owner: rename, add, seed a concept; a few dozen decisions | number of groups |

A correction in stage 4 re-clusters the whole corpus (back to stage 1). On the legal corpus that is 60,000 clauses, 64 groups, 64 naming calls, and reads on the low-cohesion groups only.

## Requirements

Every requirement below is a sentence the acceptance test can falsify.

Input
1. Accept a corpus as a folder of text files, a CSV or JSONL column, or a Postgres table and column. No labels required.
2. Accept an optional vocabulary (a list of concept names) to map against instead of discovering from scratch.
3. Embed locally by default (nomic-embed-text on Ollama). No item leaves the host unless the owner names an external endpoint.

Organization
4. Produce a hierarchy where every item is in exactly one leaf, or is listed as unfitted. Coverage is 100% and reported.
5. Name every group with one model call per group, or show the naming failure. Never backfill a name silently.
6. Report per group: size, cohesion, seam evidence (is it two concepts stuck together), representative items, ancestry.
7. Stop splitting where the geometry says a group is coherent, however coarse (emerge mode); split to depth when a vocabulary is supplied (map mode).

Confidence and spend
8. Compute cohesion for every group before any item-level model call and rank groups by it.
9. Spend per-item model calls only on the lowest-cohesion groups, up to a budget the owner sets as a read fraction.
10. Report the spend: calls made, calls avoided, and the measured quality of the propagated answers against a held-out read.

Correction
11. A rename, a new concept, or a seed item re-clusters the whole corpus; the owner sees what moved before it is published.
12. Every correction is a versioned, reversible publication with a diff.

Extraction
13. For a categorical field, extract once per sub-group and propagate; validate every propagated value with a deterministic check before it is asserted. Items that fail the check are escalated or left unresolved, never asserted.
14. For continuous or incidental attributes, route to per-item extraction and say so.

Interfaces and operations
15. Web UI on the host; HTTP API for tree, assignments, corrections, scorecard, health.
16. GET /healthz asserts what the server produced: corpus loaded, served tree covers it, naming configured.
17. Benchmark labels never influence organization, naming or routing; they only score.
18. Deterministic stages reproduce under fixed inputs, versions and seeds. A hermetic test suite gates every push.

## Quality bar

The product ships a release only when it holds the numbers it has already measured, on a held-out corpus, with the test pre-registered. The bars below are the measured state as of 2026-10-26, from the experiments in this repository.

| Metric | Measured | Bar for release | State |
| --- | --- | --- | --- |
| Confidence predicts quality (cohesion vs gold purity, Spearman) | +0.71 LEDGAR, +0.52 news, +0.74 products, 5 seeds each | >= +0.5 on an untouched corpus | holds, 3 corpora |
| High vs low confidence groups (extraction accuracy) | 98.0% vs 79.2% | the split survives a size control | holds |
| Gated extraction vs read-everything, legal (paired, -2 pt margin) | non-inferior at 40% read: 97.8% vs 97.0%, CI [-1.0, +2.8]; FAIL at 20% read, CI [-2.8, +1.8] | non-inferior at the budget the product advertises | holds at 2x fewer calls; 5x unproven |
| Gate vs random routing at equal budget | beats random on all 3 corpora at every budget | beats random | holds |
| Parity budget on news and products | no parity below 100% read | report the budget per corpus, never assume one | holds as a disclosure |
| End-to-end on tool-found concepts (legal) | routing recall 0.882, precision 0.752; 97.9% on found items; 86.4% net coverage | >= 85% coverage with a deterministic validator on propagated values | holds; validator required |
| False assertions on intruders without the validator | 88.9% of intruders get a value | 0% asserted without validation | open (requirement 13) |
| Human effort to organize 6,000 items | 20 names, 57.3% purity, about 300x fewer decisions than labeling | >= 100x at accepted quality | holds |
| Anchor validity (does a correction pull the right items) | median validity precision 0.989, CI [0.914, 1.0]; recall 0.419 | precision >= 0.90 per anchor; recall reported | precision holds; recall open |
| Anchor plus picks at zero calls vs a model reading everything | precision 0.936 vs 0.840; recall 0.47 to 0.63 vs 0.93; judge reliability bar missed by 2.5 points | the judge bar passes before the claim is made | not established |
| Seed stability of the operating point | 2.5 to 2.8 point spread across 5 seeds | spread reported with every number | holds |

The metrics that decide ship or no-ship for a release: coverage 100% accounted, non-inferiority at the advertised budget on a held-out sample, zero values asserted without the deterministic validator, and the pre-push suite green.

## Non-goals and known limits

The product does not promise a fixed saving. The call saving is a property of the corpus: large on legal clauses, absent on news posts and product listings below a full read. It reports the budget it needed instead of claiming one.

Not in scope:
- Multi-tenant hosting, enterprise authentication, or unattended high-stakes publication. The UI is loopback-only until an authorization layer exists.
- Attributes mentioned in passing. Cluster-first extraction works only when the attribute is the dominant signal of the text.
- Document-level conclusions from item-level results. A clause taxonomy does not prove contract performance.
- A universal confidence calibration. Cohesion ranks risk within a corpus and embedding model; it is calibrated per corpus family.

Published negative results that bound the design:
- Hyperbolic clustering lost to Euclidean six times, but every run used a gradient optimizer that later failed its positive control. Sarkar's construction passes on WordNet (tree distance +0.864 in two dimensions). The product clusters in Euclidean space and treats hyperbolic storage as a representation option, not a quality lever.
- Hyperbolic anomaly scoring was falsified against cosine (AUC 0.763 vs 0.814). Novelty uses Euclidean distance to the nearest leaf.
- A crystallization loop (feeding the tree back into the embedding) degrades monotonically (leaf NMI 0.522 to 0.172). The product never re-embeds on its own output.
- Name-only anchors pull in neighbours (0.37 precision); anchors are structural suggestions until a human picks the members.
- Blind whitening destroys the signal; mean-centering is the only normalization applied.

## Milestones

Each phase is defined by the test it must pass, not by a date. No date is set until the owner picks the first corpus for M1.

| Phase | Content | Gate |
| --- | --- | --- |
| M0 Today | LEDGAR demo on aurora; 64 groups over 60k clauses; the 2x operating point | /healthz ok, suite green (passed 2026-10-26) |
| M1 Any corpus in | folder, CSV/JSONL and Postgres doors; one household corpus; coverage 100% reported | a second corpus organized end to end |
| M2 Trusted extraction | deterministic validator; escalate, never assert; held-out non-inferiority | 0 false assertions at the advertised budget |
| M3 Correction loop | preview the diff; versioned publish and rollback; anchor precision measured | anchor precision >= 0.90 on a held-out corpus |

## Open decisions

- [ ] Product name. The repository, unit and register say Cohesion-Gated Taxonomy; the mechanism notes say Nomen. One name, everywhere.
- [ ] License for the public repository (owner decision cohesion-gated-taxonomy/license). Apache-2.0 carries a patent grant; MIT does not; FSL-1.1-ALv2 converts after two years.
- [ ] First corpus beyond the three measured ones. Candidates in the household data: WhatsApp messages, voice-note transcripts, Paperless documents, reel captions.
- [ ] Input doors for M1: folder, CSV/JSONL, Postgres table. Which one first follows the first corpus.
- [ ] Whether the Sarkar hyperbolic representation ships at all, or stays a research option. It passed its control but has no measured product gain.
