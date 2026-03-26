from pathlib import Path

path = Path("/workspace/action_abstraction/verl/recipe/two_policy/two_policy_trainer.py")
lines = path.read_text().splitlines()

if not any("solver_response_length: int | None = None," in line for line in lines):
    for i, line in enumerate(lines):
        if line == "        num_solver_rollouts: int,":
            lines.insert(i + 1, "        solver_response_length: int | None = None,")
            break
    else:
        raise SystemExit("signature line not found")

if not any("solver_batch.meta_info[response_length] = solver_response_length" in line for line in lines):
    for i, line in enumerate(lines):
        if "solver_batch.meta_info[temperature] = self.config.actor_rollout_ref.rollout.temperature" in line:
            lines.insert(i + 1, "            if solver_response_length is not None:")
            lines.insert(i + 2, "                solver_batch.meta_info[response_length] = solver_response_length")
            break
    else:
        raise SystemExit("solver temperature line not found")

if not any("solver_response_length = self.config.two_policy.validation_solver_response_length" in line for line in lines):
    for i, line in enumerate(lines):
        if line == "        num_solver_rollouts = self.config.two_policy.validation_num_solver_rollouts":
            lines.insert(i + 1, "        solver_response_length = self.config.two_policy.validation_solver_response_length")
            break
    else:
        raise SystemExit("validation num_solver_rollouts line not found")

if not any("solver_response_length=solver_response_length," in line for line in lines):
    for i, line in enumerate(lines):
        if line == "                num_solver_rollouts=num_solver_rollouts,":
            lines.insert(i + 1, "                solver_response_length=solver_response_length,")
            break
    else:
        raise SystemExit("validate rollout call line not found")

path.write_text("\n".join(lines) + "\n")
print("patched", path)
