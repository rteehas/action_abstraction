from pathlib import Path

# two_policy trainer
path = Path("/workspace/action_abstraction/verl/recipe/two_policy/two_policy_trainer.py")
text = path.read_text()
old = """    def _run_two_policy_rollouts(
        self,
        batch: DataProto,
        timing_raw: dict,
        do_profile: bool,
        num_abstractions: int,
        num_solver_rollouts: int,
    ):
"""
new = """    def _run_two_policy_rollouts(
        self,
        batch: DataProto,
        timing_raw: dict,
        do_profile: bool,
        num_abstractions: int,
        num_solver_rollouts: int,
        solver_response_length: int | None = None,
    ):
"""
if old not in text:
    raise SystemExit("trainer: rollout signature block not found")
text = text.replace(old, new, 1)
old = """            solver_batch.non_tensor_batch[uid] = solver_batch.non_tensor_batch[abstraction_uid].copy()
            solver_batch.meta_info[global_steps] = self.global_steps
            solver_batch.meta_info[temperature] = self.config.actor_rollout_ref.rollout.temperature

            self.checkpoint_manager.update_weights()
"""
new = """            solver_batch.non_tensor_batch[uid] = solver_batch.non_tensor_batch[abstraction_uid].copy()
            solver_batch.meta_info[global_steps] = self.global_steps
            solver_batch.meta_info[temperature] = self.config.actor_rollout_ref.rollout.temperature
            if solver_response_length is not None:
                solver_batch.meta_info[response_length] = solver_response_length

            self.checkpoint_manager.update_weights()
"""
if old not in text:
    raise SystemExit("trainer: solver meta_info block not found")
text = text.replace(old, new, 1)
old = """        num_abstractions = self.config.two_policy.validation_num_abstractions
        num_solver_rollouts = self.config.two_policy.validation_num_solver_rollouts

        problem_best_rewards = []
"""
new = """        num_abstractions = self.config.two_policy.validation_num_abstractions
        num_solver_rollouts = self.config.two_policy.validation_num_solver_rollouts
        solver_response_length = self.config.two_policy.validation_solver_response_length

        problem_best_rewards = []
"""
if old not in text:
    raise SystemExit("trainer: validate header block not found")
text = text.replace(old, new, 1)
old = """            abstraction_batch, solver_batch = self._run_two_policy_rollouts(
                batch=batch,
                timing_raw=defaultdict(float),
                do_profile=False,
                num_abstractions=num_abstractions,
                num_solver_rollouts=num_solver_rollouts,
            )
"""
new = """            abstraction_batch, solver_batch = self._run_two_policy_rollouts(
                batch=batch,
                timing_raw=defaultdict(float),
                do_profile=False,
                num_abstractions=num_abstractions,
                num_solver_rollouts=num_solver_rollouts,
                solver_response_length=solver_response_length,
            )
"""
if old not in text:
    raise SystemExit("trainer: validate rollout call block not found")
path.write_text(text.replace(old, new, 1))

# async agent loop manager
path = Path("/workspace/action_abstraction/verl/verl/experimental/agent_loop/agent_loop.py")
text = path.read_text()
old = """        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
"""
new = """        config = self.rollout_config
        response_length = int(batch.meta_info.get(\"response_length\", config.response_length))
        sampling_params = dict(
            temperature=batch.meta_info.get(\"temperature\", config.temperature),
            top_p=batch.meta_info.get(\"top_p\", config.top_p),
            top_k=batch.meta_info.get(\"top_k\", config.top_k),
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
            max_tokens=response_length,
        )
"""
if old not in text:
    raise SystemExit("agent_loop: sampling params block not found")
text = text.replace(old, new, 1)
old = """                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
"""
new = """                    self._run_agent_loop(
                        sampling_params,
                        trajectory_info[i],
                        trace=trace_this_sample,
                        response_length=response_length,
                        **kwargs,
                    )
"""
if old not in text:
    raise SystemExit("agent_loop: task launch block not found")
text = text.replace(old, new, 1)
old = """        output.extra_fields[\"raw_prompt\"] = kwargs[\"raw_prompt\"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.
"""
new = """        output.extra_fields[\"raw_prompt\"] = kwargs[\"raw_prompt\"]
        max_response_length = int(kwargs.get(\"response_length\", self.rollout_config.response_length))

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.
"""
if old not in text:
    raise SystemExit("agent_loop: postprocess header block not found")
text = text.replace(old, new, 1)
text = text.replace("max_length=self.rollout_config.response_length", "max_length=max_response_length")
text = text.replace(
    "pad_size = self.rollout_config.response_length - len(output.response_logprobs)",
    "pad_size = max_response_length - len(output.response_logprobs)",
)
path.write_text(text)

# full launchers
for path_str in [
    "/workspace/action_abstraction/verl/examples/grpo_trainer/run_qwen_two_policy_full.sh",
    "/workspace/action_abstraction/verl/examples/grpo_trainer/run_qwen_two_policy_full_dapo.sh",
]:
    path = Path(path_str)
    text = path.read_text()
    old = """ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
"""
    new = """ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}
VALIDATION_SOLVER_MAX_RESP_LEN=${VALIDATION_SOLVER_MAX_RESP_LEN:-32768}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
"""
    if old not in text:
        raise SystemExit(f"launcher: length env block not found in {path}")
    text = text.replace(old, new, 1)
    old = """  two_policy.validation_num_abstractions=4
  two_policy.validation_num_solver_rollouts=4
  trainer.critic_warmup=0
"""
    new = """  two_policy.validation_num_abstractions=4
  two_policy.validation_num_solver_rollouts=4
  two_policy.validation_solver_response_length=\"${VALIDATION_SOLVER_MAX_RESP_LEN}\"
  trainer.critic_warmup=0
"""
    if old not in text:
        raise SystemExit(f"launcher: validation block not found in {path}")
    path.write_text(text.replace(old, new, 1))
