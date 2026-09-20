# Task 4: following a safer escape after placement

The 300-pair placement-guard comparison on world seeds 26000–26299 recorded
52 self-kills with the guard off and 26 with it on. Mean score was 3.327 versus
3.483. The exploratory paired bootstrap interval supports a reduction in
self-kills, but does not establish better score or win rate. The baseline is
the recovered Safety v2 episode-4000 checkpoint with the placement guard on.
Full records are in `task4_guard_confirmation/`.

Of the 26 remaining self-kills, 25 included a riskier escape choice while a
predicted safer alternative existed; 22 included an intercepted movement.
These overlapping observations motivate this experiment, not a causal claim.

`AGENT_ESCAPE_COLLISION_GUARD=1` prefers movement or WAIT actions whose escape
avoids tiles opponents can reach first, when already in an observed bomb's
blast path. It applies to observed friendly and enemy bombs because game-state
bombs do not expose ownership. When no such route is found, it retains the
existing movement choices. Bomb-placement screening remains independent.
The new option defaults to **off**, preserving the evaluated baseline.

The implementation and three regression tests were added without executing
commands, per the user's preference. Run these from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

If the tests pass, compare escape guard off/on with the placement guard on in
both arms and the same unchanged Safety v2 checkpoint. No retraining is needed:

```sh
.venv/bin/python -u experiments/compare_task4_guard.py --guard-kind escape --seed-start 27000 --n-rounds 300 --out-dir experiments/task4_escape_guard_confirmation
.venv/bin/python experiments/analyze_task4_guard.py experiments/task4_escape_guard_confirmation
```

The output directory must be new. The supplied opponents retain unseeded
randomness. This experiment has not been run, and no performance gain is yet
established. Do not use these evaluation seeds for training.
