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

PREMISE: supported
