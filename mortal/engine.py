import json
import logging
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *
from model import Brain, DQN, CategoricalPolicy

def resolve_amp_dtype(cfg_control):
    dtype_str = cfg_control.get('amp_dtype', 'float16')
    return torch.bfloat16 if dtype_str == 'bfloat16' else torch.float16

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        amp_dtype = None,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        name = 'NoName',
        boltzmann_epsilon = 0,
        boltzmann_temp = 1,
        top_p = 1,
        weights_dtype = None,
        bucket_batch = False,
        buckets = None,
        use_cuda_graphs = False,
        graph_max_batch = 512,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.amp_dtype = amp_dtype or torch.float16
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

        # cast weights once instead of letting autocast re-cast all of them on
        # every react_batch call; this mutates the modules passed in, so never
        # enable it for modules that are still being trained
        self.infer_dtype = None
        if weights_dtype in ('bf16', 'bfloat16'):
            self.infer_dtype = torch.bfloat16
            self.brain = self.brain.to(torch.bfloat16)
            self.dqn = self.dqn.to(torch.bfloat16)
            self.enable_amp = False

        self.bucket_batch = bucket_batch and self.device.type == 'cuda'
        self.buckets = sorted(buckets) if buckets else [64, 128, 256, 512, 1024, 2048]
        # works with autocast too: capture happens under cache_enabled=False
        # (see _GraphRunner); version 1 is excluded because of its sampling
        # latent path
        self.use_cuda_graphs = (
            use_cuda_graphs
            and self.bucket_batch
            and self.version != 1
        )
        self.graph_max_batch = graph_max_batch
        self._staging_bufs = {}
        self._graphs = {}

        self.react_calls = 0
        self.react_samples = 0
        self.react_max_batch = 0

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        batch_size = len(obs)
        self.react_calls += 1
        self.react_samples += batch_size
        self.react_max_batch = max(self.react_max_batch, batch_size)

        if self.bucket_batch and invisible_obs is None:
            q_out, masks = self._forward_bucketed(obs, masks, batch_size)
        else:
            obs = torch.as_tensor(np.stack(obs, axis=0), device=self.device)
            masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
            if invisible_obs is not None:
                invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
            q_out = self._forward(obs, masks, invisible_obs)
        q_out = q_out.float()

        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

    def _forward(self, obs, masks, invisible_obs=None):
        if self.infer_dtype is not None and obs.dtype != self.infer_dtype:
            obs = obs.to(self.infer_dtype)
            if invisible_obs is not None:
                invisible_obs = invisible_obs.to(self.infer_dtype)

        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks)
            case 2 | 3 | 4:
                phi = self.brain(obs)
                q_out = self.dqn(phi, masks)
        return q_out

    def _forward_bucketed(self, obs_list, mask_list, batch_size):
        target = next((b for b in self.buckets if b >= batch_size), None)
        if target is None:
            target = 1 << (batch_size - 1).bit_length()

        obs_buf = self._staging('obs', target, obs_list[0].shape, torch.float32)
        mask_buf = self._staging('mask', target, mask_list[0].shape, torch.bool)
        np.stack(obs_list, axis=0, out=obs_buf.numpy()[:batch_size])
        np.stack(mask_list, axis=0, out=mask_buf.numpy()[:batch_size])
        if target > batch_size:
            # padded rows keep action 45 legal so that the softmax in the
            # sampling path stays finite; they are sliced off below
            obs_buf[batch_size:].zero_()
            mask_buf[batch_size:].zero_()
            mask_buf[batch_size:, 45] = True

        masks = mask_buf.to(self.device, non_blocking=True)
        q_out = None
        if self.use_cuda_graphs and target <= self.graph_max_batch:
            q_out = self._replay_graph(target, obs_buf, masks)
        if q_out is None:
            obs = obs_buf.to(self.device, non_blocking=True)
            q_out = self._forward(obs, masks)
        return q_out[:batch_size], masks[:batch_size]

    def _staging(self, name, rows, item_shape, dtype):
        # reusing the buffer across calls is safe: the tolist() at the end of
        # the previous react_batch synchronizes the device before the next
        # call writes to it
        key = (name, rows)
        buf = self._staging_bufs.get(key)
        if buf is None or buf.shape[1:] != torch.Size(item_shape):
            buf = torch.empty((rows, *item_shape), dtype=dtype, pin_memory=True)
            self._staging_bufs[key] = buf
        return buf

    def _replay_graph(self, target, obs_buf, masks_dev):
        runner = self._graphs.get(target)
        if runner is None:
            try:
                runner = _GraphRunner(self, target, obs_buf.shape[1:], masks_dev.shape[1:])
            except Exception as e:
                logging.warning(f'CUDA graph capture failed (batch={target}), falling back to eager: {e}')
                self.use_cuda_graphs = False
                self._graphs.clear()
                return None
            self._graphs[target] = runner
        return runner.run(obs_buf, masks_dev)

    def take_react_stats(self):
        calls, samples, max_batch = self.react_calls, self.react_samples, self.react_max_batch
        self.react_calls = self.react_samples = self.react_max_batch = 0
        return {
            'calls': calls,
            'samples': samples,
            'avg_batch': samples / calls if calls > 0 else 0.,
            'max_batch': max_batch,
        }

class _GraphRunner:
    def __init__(self, engine, batch_size, obs_shape, mask_shape):
        device = engine.device
        self.static_obs = torch.zeros((batch_size, *obs_shape), dtype=torch.float32, device=device)
        self.static_masks = torch.zeros((batch_size, *mask_shape), dtype=torch.bool, device=device)
        self.static_masks[:, 45] = True # keep the softmax finite during warmup/capture

        # capturing under autocast requires its weight cache to be disabled;
        # the cast ops then get recorded into the graph itself
        def amp_ctx():
            return torch.autocast(
                device.type,
                dtype=engine.amp_dtype,
                enabled=engine.enable_amp,
                cache_enabled=False,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                with amp_ctx():
                    engine._forward(self.static_obs, self.static_masks)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with amp_ctx():
                self.static_q = engine._forward(self.static_obs, self.static_masks)

    def run(self, obs_src, masks_src):
        self.static_obs.copy_(obs_src, non_blocking=True)
        self.static_masks.copy_(masks_src, non_blocking=True)
        self.graph.replay()
        return self.static_q

def build_engine_from_state(
    state, *,
    device,
    enable_compile=False,
    enable_amp=True,
    enable_rule_based_agari_guard=True,
    name='baseline',
    weights_dtype=None,
    bucket_batch=False,
    buckets=None,
    use_cuda_graphs=False,
    graph_max_batch=512,
):
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']

    brain_state = state['mortal']
    has_bn_stats = any('running_mean' in k for k in brain_state.keys())
    norm = 'bn' if has_bn_stats else 'gn'

    if 'policy_net' in state:
        head_state = state['policy_net']
    elif 'current_dqn' in state:
        head_state = state['current_dqn']
    else:
        raise KeyError("checkpoint has neither 'policy_net' nor 'current_dqn'")

    is_policy = ('fc1.weight' in head_state) and ('fc2.weight' in head_state)
    if is_policy and version == 1:
        raise ValueError('CategoricalPolicy does not support version 1 (latent dim mismatch)')

    brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks, norm=norm).eval()
    head = CategoricalPolicy().eval() if is_policy else DQN(version=version).eval()
    brain.load_state_dict(brain_state)
    head.load_state_dict(head_state)

    if enable_compile:
        brain.compile()
        head.compile()

    head_kind = 'policy' if is_policy else 'dqn'
    engine = MortalEngine(
        brain,
        head,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=enable_amp,
        amp_dtype=resolve_amp_dtype(cfg['control']),
        enable_rule_based_agari_guard=enable_rule_based_agari_guard,
        name=name,
        weights_dtype=weights_dtype,
        bucket_batch=bucket_batch,
        buckets=buckets,
        use_cuda_graphs=use_cuda_graphs,
        graph_max_batch=graph_max_batch,
    )
    return engine, {'head_kind': head_kind, 'norm': norm, 'version': version}

def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass
