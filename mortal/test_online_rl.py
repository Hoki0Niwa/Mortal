import unittest

import numpy as np
import torch
from torch import nn

from dataloader import calculate_mc_targets
from engine import MortalEngine, build_engine_from_state
from model import Brain, BootstrappedDQN, DQN, load_dqn_state_compat
from prepare_online_seed import prepare_state


class BootstrappedDQNTest(unittest.TestCase):
    def test_legacy_head_is_replicated_and_prior_is_fixed(self):
        torch.manual_seed(7)
        legacy = DQN(version=4)
        ensemble = BootstrappedDQN(version=4, num_heads=3, prior_scale=0.2)
        self.assertEqual(load_dqn_state_compat(ensemble, legacy.state_dict()), 'legacy-replicated')

        for head in ensemble.heads:
            for expected, actual in zip(legacy.parameters(), head.parameters()):
                torch.testing.assert_close(expected, actual)
        self.assertTrue(all(not p.requires_grad for p in ensemble.prior_heads.parameters()))

    def test_head_selection_is_per_sample(self):
        torch.manual_seed(11)
        ensemble = BootstrappedDQN(version=4, num_heads=3, prior_scale=0.2)
        phi = torch.randn(4, 1024)
        mask = torch.ones(4, 46, dtype=torch.bool)
        heads = ensemble.forward_heads(phi, mask)
        ids = torch.tensor([0, 2, 1, 0])
        selected = ensemble(phi, mask, head_ids=ids)
        torch.testing.assert_close(selected, heads[torch.arange(4), ids])
        torch.testing.assert_close(ensemble(phi, mask), heads.mean(dim=1))

    def test_engine_keeps_one_head_for_each_game(self):
        ensemble = BootstrappedDQN(version=4, num_heads=3, prior_scale=0)
        with torch.no_grad():
            for idx, head in enumerate(ensemble.heads):
                head.net.weight.zero_()
                head.net.bias.zero_()
                head.net.bias[1 + idx] = 1
        engine = MortalEngine(
            nn.Identity(),
            ensemble,
            is_oracle=False,
            version=4,
            ensemble_mode='sample_per_game',
            head_seed=19,
        )
        obs = np.zeros((1024,), dtype=np.float32)
        mask = np.ones((46,), dtype=np.bool_)
        actions, *_ = engine.react_batch_with_indices(
            [obs, obs, obs], [mask, mask, mask], None, [7, 7, 9]
        )
        self.assertEqual(actions[0], actions[1])
        repeated, *_ = engine.react_batch_with_indices([obs], [mask], None, [7])
        self.assertEqual(repeated[0], actions[0])
        stats = engine.take_react_stats()
        self.assertEqual(sum(stats['head_counts']), 2)
        with self.assertRaises(Exception):
            engine.react_batch([obs], [mask], None)

    def test_checkpoint_factory_restores_bootstrapped_head(self):
        brain = Brain(version=4, conv_channels=32, num_blocks=1)
        dqn = BootstrappedDQN(version=4, num_heads=3, prior_scale=0.2)
        state = {
            'mortal': brain.state_dict(),
            'current_dqn': dqn.state_dict(),
            'config': {
                'control': {'version': 4, 'amp_dtype': 'float16'},
                'resnet': {'conv_channels': 32, 'num_blocks': 1},
                'online_rl': {
                    'bootstrapped_dqn': True,
                    'num_heads': 3,
                    'prior_scale': 0.2,
                },
            },
        }
        engine, info = build_engine_from_state(state, device=torch.device('cpu'))
        self.assertIsInstance(engine.dqn, BootstrappedDQN)
        self.assertEqual(engine.dqn.num_heads, 3)
        self.assertEqual(info['head_kind'], 'bootstrapped_dqn')


class HanchanReturnTest(unittest.TestCase):
    def test_hanchan_return_accumulates_future_kyoku_deltas(self):
        at_kyoku = np.array([0, 0, 1, 1, 2])
        dones = np.array([False, True, False, True, True])
        apply_gamma = np.array([True, True, True, False, True])
        rewards = np.array([0.5, -1.0, 2.0])

        target, steps = calculate_mc_targets(
            at_kyoku, dones, apply_gamma, rewards, 'hanchan_return'
        )
        np.testing.assert_allclose(target, [1.5, 1.5, 1.0, 1.0, 2.0])
        np.testing.assert_array_equal(steps, [3, 2, 1, 0, 0])

    def test_kyoku_delta_remains_compatible(self):
        at_kyoku = np.array([0, 0, 1])
        dones = np.array([False, True, True])
        apply_gamma = np.array([True, True, True])
        rewards = np.array([0.5, -1.0])
        target, steps = calculate_mc_targets(
            at_kyoku, dones, apply_gamma, rewards, 'kyoku_delta'
        )
        np.testing.assert_allclose(target, [0.5, 0.5, -1.0])
        np.testing.assert_array_equal(steps, [1, 0, 0])


class OnlineSeedTest(unittest.TestCase):
    def test_progress_is_reset_without_changing_weights(self):
        weight = torch.randn(2, 3)
        state = {
            'mortal': {'weight': weight},
            'current_dqn': {'weight': weight.clone()},
            'aux_net': {'weight': weight.clone()},
            'config': {'control': {'online': False}},
            'steps': 950000,
            'best_perf': {'avg_rank': 2.48, 'avg_pt': 1.2},
            'top_checkpoints': [{'filepath': 'old-machine-path'}],
        }
        result = prepare_state(state, 'main.pth', 'abc123')
        self.assertEqual(result['steps'], 0)
        self.assertEqual(result['top_checkpoints'], [])
        self.assertEqual(result['best_perf'], {'avg_rank': 4.0, 'avg_pt': -135.0})
        self.assertIs(result['mortal']['weight'], weight)
        self.assertEqual(result['online_seed']['source_sha256'], 'abc123')


if __name__ == '__main__':
    unittest.main()
