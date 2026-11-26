# catalog-retrieval-hyperbolic-01: premise check, the gold labels mark items an agent should read

Premise: each task's gold items (the skill the agent invoked; the decisions the brief cited) are items an agent doing
that task should read, and the task text does not name them.

Label source (never a model): skill gold = a real Skill tool call in the transcript after the task text (82 of 125
skill tasks also appear in governance.skill_invocation); decision gold = owner_decision:<subject> citations written
by the brief's author. Gold names are masked in the text before any arm runs.

Checks:
- Automated leak scan of key v2 (432 tasks): 0 tasks with a gold skill name (hyphen or space form) or a gold decision
  subject left in the text.
- Hand check 1 (key v1, 20 tasks, seed handcheck): 18 plausible; 2 were not tasks (a harness task-notification, which
  is an agent's report, and a batch-judge prompt whose citations sit inside the judged records). Both forms are
  excluded in key v2 (0 left), before any test scoring.
- Hand check 2 (key v2, 20 tasks, seed handcheck-v2): 19 plausible, 1 weak (a skill call whose preceding user message
  is a follow-up, not the request that led to the skill).
- Estimated label noise: about 5% weak labels. It lowers every arm and power; it does not favour one arm.
- Known incompleteness: gold holds only the items that were used or cited. An unused relevant item counts against
  every arm that ranks it high. The agent failures of 2026-11-25 (ponytail-workflow, the leak register, Rule A not
  read) are of this kind, so recall is measured against what agents did, not against all that was relevant.
- 44% of the test units (64 of 146) are automated security-review prompts over a diff whose code cites the decision.
  They are real agent tasks; the breakdown by task type is reported.

Hand checks read task texts and labels only, never an arm's output on test.

## Amendment v2 checks: key v3 (2026-11-25, before any test-split run; review findings F2 and F12)

Key v3 = key v2 plus two rules (build_key.py): (1) the distinctive fragments of EVERY catalog decision subject (the
slug without its domain, with and without its date, in hyphen, underscore and space forms; only slugs of two or more
parts) and every 8-digit date token are masked, in every task, so a mask never marks the gold; (2) a task whose token
set has Jaccard >= 0.75 with an earlier kept task is dropped. The flag gold_fragment records whether the text held a
fragment or the date of its own gold BEFORE this masking.

- Key v3: 424 tasks (dev 115, test 309); tasks.jsonl sha256 71f49d73...; the split seed and rule are unchanged.
- Automated leak scan of key v3 (all 424 tasks): 0 gold skill names, 0 full gold subjects, 0 slug fragments of the
  gold. 7 tasks (5 test, 2 dev) keep the 8-digit date of their gold inside a longer token (a file or branch name),
  which the date rule does not split. A date is shared by many decisions, so it is a weak cue; these tasks are in
  the gold_fragment stratum.
- gold_fragment stratum (reported for every arm): 84 of 424 tasks (dev 30, test 54); 25 of the 144 test units.
- Mask density: 306 of 424 tasks hold one or more [decision] marks (median 1, largest 7); 34 hold a [date] mark.
- Hand check 3 (key v3, 20 tasks, seed handcheck-v3, texts and labels only): 19 plausible, 1 weak (a skill call
  after a short follow-up message). In 3 of the 20, the mask also removed a generic phrase that equals another
  decision's slug (for example a product phrase). This over-masking removes lexical cues for every arm; it cannot
  point to the gold.
- Label noise estimate stays at about 5% weak labels.
- Limit that stays (review finding F7): the labels come from what an author cited or an agent invoked. A security
  review gold is the decision that the diff cites; a brief's gold is the decision its author had in mind. Both
  measure a reverse lookup from a paraphrase. The skill_call type (25 test units) is the nearest proxy for "the
  agent needs X and does not know it". The design names this as an alternative explanation for every contrast.

PREMISE: supported
