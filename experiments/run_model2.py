"""Run an isolated CNN experiment, then evaluate best and latest checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--eval-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--snapshot-from", type=Path,
                        help="Use a previous frozen source tree, with the current checkpoint writer")
    args = parser.parse_args()
    if min(args.rounds, args.eval_rounds) < 1:
        parser.error("Round counts must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"cnn_task2_seed{args.seed}_{stamp}"
    output = ROOT / "results" / "model2_runs" / run_name
    source = output / "source"
    source.mkdir(parents=True, exist_ok=False)
    hashes = {}
    source_root = args.snapshot_from.resolve() if args.snapshot_from else ROOT
    files = list(source_root.glob("*.py")) + list((source_root / "agent_code").rglob("*.py"))
    for original in files:
        relative = original.relative_to(source_root)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == "agent_code/our_agent/model_net.py":
            original = ROOT / relative
        content = original.read_bytes()
        destination.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    shutil.copytree(source_root / "assets", source / "assets")
    (source / "logs").mkdir()
    (source / "results").mkdir()
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    env = os.environ.copy()
    env.pop("AGENT_INIT_CHECKPOINT", None)
    env.update(AGENT_MODEL="net", AGENT_DEVICE="cpu", AGENT_PRESET="task2",
               AGENT_RUN=run_name, AGENT_SEED=str(args.seed))
    if args.init_checkpoint:
        checkpoint = args.init_checkpoint.resolve(strict=True)
        copied = output / "initial_checkpoint.pt"
        shutil.copy2(checkpoint, copied)
        env["AGENT_INIT_CHECKPOINT"] = str(copied)
    status = dict(run=run_name, pid=os.getpid(), started_utc=stamp,
                  rounds=args.rounds, eval_rounds=args.eval_rounds, seed=args.seed,
                  device="cpu", source=str(source), phase="starting")
    if args.init_checkpoint:
        status.update(initial_checkpoint=str(checkpoint),
                      restart_mode="weight transfer; replay and schedules restart")

    def save_status(**updates):
        status.update(updates)
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(output / "status.json")

    def execute(command, label, process_env):
        save_status(phase=label, command=command)
        with (output / f"{label}.stdout.log").open("w") as stdout, \
                (output / f"{label}.stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, cwd=source, env=process_env,
                                       stdout=stdout, stderr=stderr)
            save_status(child_pid=process.pid)
            code = process.wait()
        if code:
            raise RuntimeError(f"{label} failed with exit code {code}; inspect its logs")

    save_status()
    print(output, flush=True)
    try:
        execute([sys.executable, "main.py", "play", "--agents", "our_agent",
                 "--scenario", "loot-crate", "--train", "1", "--no-gui",
                 "--n-rounds", str(args.rounds), "--seed", str(args.seed),
                 "--save-stats", str(output / "training_stats.json")], "training", env)
        weights = source / "agent_code" / "our_agent" / "weights"
        export = output / "checkpoints"
        export.mkdir()
        for candidate, suffix in (("best", "_best"), ("latest", "")):
            checkpoint = weights / f"q_net_{run_name}{suffix}.pt"
            if not checkpoint.exists():
                if candidate == "best" and args.rounds < 100:
                    continue
                raise FileNotFoundError(checkpoint)
            shutil.copy2(checkpoint, export / checkpoint.name)
            evaluation_name = f"{run_name}_eval_{candidate}"
            shutil.copy2(checkpoint, weights / f"q_net_{evaluation_name}.pt")
            evaluation_env = dict(env, AGENT_RUN=evaluation_name)
            execute([sys.executable, "evaluate.py", "--agent", "our_agent",
                     "--task", "2", "--n-rounds", str(args.eval_rounds),
                     "--out", str(output / f"eval_{candidate}.csv"),
                     "--keep-stats-files"], f"evaluate_{candidate}", evaluation_env)
            # Preserve each candidate's raw JSON before the next one reuses names.
            shutil.copytree(source / "results", output / f"raw_{candidate}")
        shutil.copy2(source / "agent_code" / "our_agent" / f"training_net_{run_name}.csv",
                     output / "training.csv")
        save_status(phase="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        save_status(phase="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
