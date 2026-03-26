from pathlib import Path

path = Path('/workspace/action_abstraction/verl/recipe/two_policy/two_policy_trainer.py')
text = path.read_text()
old = """    def _validate(self, merged: bool = False):
        del merged
        num_abstractions = self.config.two_policy.validation_num_abstractions
        num_solver_rollouts = self.config.two_policy.validation_num_solver_rollouts

        problem_best_rewards = []
        problem_mean_rewards = []
        problem_best_accs = []
        problem_mean_accs = []
        abstraction_validities = []

        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            abstraction_batch, solver_batch = self._run_two_policy_rollouts(
                batch=batch,
                timing_raw=defaultdict(float),
                do_profile=False,
                num_abstractions=num_abstractions,
                num_solver_rollouts=num_solver_rollouts,
            )
            abstraction_validities.extend(abstraction_batch.non_tensor_batch['abstraction_valid'].astype(np.float32).tolist())

            problem_uids = abstraction_batch.non_tensor_batch['problem_uid']
            unique_problem_uids = []
            seen = set()
            for uid in problem_uids:
                if uid not in seen:
                    unique_problem_uids.append(uid)
                    seen.add(uid)

            if solver_batch is None or len(solver_batch) == 0:
                problem_best_rewards.extend([0.0] * len(unique_problem_uids))
                problem_mean_rewards.extend([0.0] * len(unique_problem_uids))
                continue

            reward_by_problem = defaultdict(list)
            acc_by_problem = defaultdict(list)
            for idx, problem_uid in enumerate(solver_batch.non_tensor_batch['problem_uid']):
                reward_by_problem[problem_uid].append(float(solver_batch.non_tensor_batch['seq_final_reward'][idx]))
                if 'acc' in solver_batch.non_tensor_batch:
                    acc_by_problem[problem_uid].append(float(solver_batch.non_tensor_batch['acc'][idx]))

            for problem_uid in unique_problem_uids:
                rewards = reward_by_problem.get(problem_uid, [0.0])
                problem_best_rewards.append(float(np.max(rewards)))
                problem_mean_rewards.append(float(np.mean(rewards)))
                if acc_by_problem:
                    accs = acc_by_problem.get(problem_uid, [0.0])
                    problem_best_accs.append(float(np.max(accs)))
                    problem_mean_accs.append(float(np.mean(accs)))

        metrics = {
            'val/abstraction_valid_rate': float(np.mean(abstraction_validities)) if abstraction_validities else 0.0,
            'val/problem_best_reward_mean': float(np.mean(problem_best_rewards)) if problem_best_rewards else 0.0,
            'val/problem_mean_reward_mean': float(np.mean(problem_mean_rewards)) if problem_mean_rewards else 0.0,
        }
        if problem_best_accs:
            metrics['val/problem_best_acc_mean'] = float(np.mean(problem_best_accs))
            metrics['val/problem_mean_acc_mean'] = float(np.mean(problem_mean_accs))
        return metrics
"""
new = """    def _validate(self, merged: bool = False):
        del merged
        num_abstractions = self.config.two_policy.validation_num_abstractions
        num_solver_rollouts = self.config.two_policy.validation_num_solver_rollouts

        problem_best_rewards = []
        problem_mean_rewards = []
        problem_best_accs = []
        problem_mean_accs = []
        abstraction_validities = []

        source_problem_best_rewards = defaultdict(list)
        source_problem_mean_rewards = defaultdict(list)
        source_problem_best_accs = defaultdict(list)
        source_problem_mean_accs = defaultdict(list)
        source_abstraction_validities = defaultdict(list)
        source_names = set()

        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            abstraction_batch, solver_batch = self._run_two_policy_rollouts(
                batch=batch,
                timing_raw=defaultdict(float),
                do_profile=False,
                num_abstractions=num_abstractions,
                num_solver_rollouts=num_solver_rollouts,
            )
            abstraction_valid_array = abstraction_batch.non_tensor_batch['abstraction_valid'].astype(np.float32)
            abstraction_validities.extend(abstraction_valid_array.tolist())

            batch_sources = abstraction_batch.non_tensor_batch.get('data_source')
            if batch_sources is None:
                batch_sources = np.asarray(['unknown'] * len(abstraction_batch), dtype=object)
            normalized_sources = [str(source).strip().lower().replace(' ', '_') for source in batch_sources]
            source_names.update(normalized_sources)
            for source, is_valid in zip(normalized_sources, abstraction_valid_array, strict=True):
                source_abstraction_validities[source].append(float(is_valid))

            problem_uids = abstraction_batch.non_tensor_batch['problem_uid']
            unique_problem_uids = []
            problem_source = {}
            seen = set()
            for idx, uid in enumerate(problem_uids):
                if uid not in seen:
                    unique_problem_uids.append(uid)
                    problem_source[uid] = normalized_sources[idx]
                    seen.add(uid)

            if solver_batch is None or len(solver_batch) == 0:
                for problem_uid in unique_problem_uids:
                    source = problem_source[problem_uid]
                    problem_best_rewards.append(0.0)
                    problem_mean_rewards.append(0.0)
                    source_problem_best_rewards[source].append(0.0)
                    source_problem_mean_rewards[source].append(0.0)
                continue

            reward_by_problem = defaultdict(list)
            acc_by_problem = defaultdict(list)
            for idx, problem_uid in enumerate(solver_batch.non_tensor_batch['problem_uid']):
                reward_by_problem[problem_uid].append(float(solver_batch.non_tensor_batch['seq_final_reward'][idx]))
                if 'acc' in solver_batch.non_tensor_batch:
                    acc_by_problem[problem_uid].append(float(solver_batch.non_tensor_batch['acc'][idx]))

            for problem_uid in unique_problem_uids:
                source = problem_source[problem_uid]
                rewards = reward_by_problem.get(problem_uid, [0.0])
                best_reward = float(np.max(rewards))
                mean_reward = float(np.mean(rewards))
                problem_best_rewards.append(best_reward)
                problem_mean_rewards.append(mean_reward)
                source_problem_best_rewards[source].append(best_reward)
                source_problem_mean_rewards[source].append(mean_reward)
                if acc_by_problem:
                    accs = acc_by_problem.get(problem_uid, [0.0])
                    best_acc = float(np.max(accs))
                    mean_acc = float(np.mean(accs))
                    problem_best_accs.append(best_acc)
                    problem_mean_accs.append(mean_acc)
                    source_problem_best_accs[source].append(best_acc)
                    source_problem_mean_accs[source].append(mean_acc)

        metrics = {
            'val/abstraction_valid_rate': float(np.mean(abstraction_validities)) if abstraction_validities else 0.0,
            'val/problem_best_reward_mean': float(np.mean(problem_best_rewards)) if problem_best_rewards else 0.0,
            'val/problem_mean_reward_mean': float(np.mean(problem_mean_rewards)) if problem_mean_rewards else 0.0,
        }
        if problem_best_accs:
            metrics['val/problem_best_acc_mean'] = float(np.mean(problem_best_accs))
            metrics['val/problem_mean_acc_mean'] = float(np.mean(problem_mean_accs))

        for source in sorted(source_names):
            metrics[f'val/{source}/abstraction_valid_rate'] = (
                float(np.mean(source_abstraction_validities[source])) if source_abstraction_validities[source] else 0.0
            )
            metrics[f'val/{source}/problem_best_reward_mean'] = (
                float(np.mean(source_problem_best_rewards[source])) if source_problem_best_rewards[source] else 0.0
            )
            metrics[f'val/{source}/problem_mean_reward_mean'] = (
                float(np.mean(source_problem_mean_rewards[source])) if source_problem_mean_rewards[source] else 0.0
            )
            if problem_best_accs:
                metrics[f'val/{source}/problem_best_acc_mean'] = (
                    float(np.mean(source_problem_best_accs[source])) if source_problem_best_accs[source] else 0.0
                )
                metrics[f'val/{source}/problem_mean_acc_mean'] = (
                    float(np.mean(source_problem_mean_accs[source])) if source_problem_mean_accs[source] else 0.0
                )
        return metrics
"""
if old not in text:
    raise SystemExit('validate block not found')
path.write_text(text.replace(old, new, 1))
