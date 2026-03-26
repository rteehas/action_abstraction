from pathlib import Path

path = Path("/workspace/action_abstraction/verl/recipe/two_policy/two_policy_trainer.py")
text = path.read_text()

if "solver_response_length: int | None = None," not in text:
    text = text.replace(
        "        num_solver_rollouts: int,\n    ):\n",
        "        num_solver_rollouts: int,\n        solver_response_length: int | None = None,\n    ):\n",
        1,
    )

needle = "            solver_batch.meta_info[temperature] = self.config.actor_rollout_ref.rollout.temperature\n"
insert = "            if solver_response_length is not None:\n                solver_batch.meta_info[response_length] = solver_response_length\n"
if "solver_batch.meta_info[response_length] = solver_response_length" not in text:
    if needle not in text:
        raise SystemExit("temperature line not found")
    text = text.replace(needle, needle + insert, 1)

needle = "        num_solver_rollouts = self.config.two_policy.validation_num_solver_rollouts\n"
insert = "        solver_response_length = self.config.two_policy.validation_solver_response_length\n"
if insert not in text:
    if needle not in text:
        raise SystemExit("validate rollout count line not found")
    text = text.replace(needle, needle + insert, 1)

old = "                num_solver_rollouts=num_solver_rollouts,\n            )\n"
new = "                num_solver_rollouts=num_solver_rollouts,\n                solver_response_length=solver_response_length,\n            )\n"
if "solver_response_length=solver_response_length" not in text:
    if old not in text:
        raise SystemExit("validate rollout call block not found")
    text = text.replace(old, new, 1)

path.write_text(text)
print("patched", path)
