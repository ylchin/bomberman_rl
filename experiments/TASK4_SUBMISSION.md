# Task 4 submission

Freeze the existing symmetry-averaged Safety v2 model. Default evaluation loads
`agent_code/our_agent/weights/q_linear_safety_v2_symavg.pkl`, SHA256
`c5671d5e806e8d563fd1b21832eaee8d01a54e86ba592a4a5898273c553d973f`.
Survival filtering, bomb collision guard and escape collision guard are on.
Optimistic fallback and coin preference are off. No retraining is required.
This is the same checkpoint and guard combination previously evaluated; only
the default checkpoint path and escape-guard default were changed for submission.

## Run locally before submitting

The assistant has not run these tests or games. In the repository root, clear
experimental overrides in this terminal so the checks exercise submission defaults:

```sh
unset AGENT_MODEL AGENT_RUN AGENT_PRESET AGENT_EVAL_CHECKPOINT AGENT_SEED
unset AGENT_SURVIVAL_FILTER AGENT_BOMB_COLLISION_GUARD AGENT_ESCAPE_COLLISION_GUARD
unset AGENT_OPTIMISTIC_FALLBACK AGENT_COIN_PREFERENCE

.venv/bin/python -m unittest discover -s tests -v
```

The new submission tests verify no-environment defaults, the checkpoint hash
and that its schema can be loaded. If the tests pass, check normal entry-point
loading for five games:

```sh
.venv/bin/python main.py play \
  --agents our_agent rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --no-gui --n-rounds 5 --seed 37000
```

Then one final 100-game validation, with no `--checkpoint` override:

```sh
.venv/bin/python evaluate.py \
  --agent our_agent --task 4 --seed-start 37100 --n-rounds 100 \
  --out experiments/task4_submission_validation.csv
```

Do not tune the model to this final batch. Its role is to check the actual
submission configuration and provide a final reported sample. Earlier results
remain separate; world seeds do not seed opponent randomness. If submission
requires a particular container, also check this entry point in that required
environment. The repository does not specify the course's upload/archive layout.

## Include the artifact

The named checkpoint now has an explicit `.gitignore` exception; other training
checkpoints remain ignored. For Git-based delivery, stage the submission changes
and checkpoint explicitly after validation:

```sh
git add .gitignore agent_code/our_agent/config.py \
  agent_code/our_agent/weights/q_linear_safety_v2_symavg.pkl \
  tests/test_task4_submission.py experiments/TASK4_SUBMISSION.md README.md
git diff --cached --stat
```

Review any changes already staged before committing. For an archive submission,
include the agent's Python modules and the named checkpoint at the same relative
path. Do not rely on shell environment variables being available to the grader.
Keep the frozen model bytes unchanged. Follow the course's required archive and
report format; no archive layout has been assumed here.
