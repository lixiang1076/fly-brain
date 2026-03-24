#!/usr/bin/env python3
"""
fast_learning.py — Fast dopamine learning using post-hoc MBON modulation.

Instead of modifying Brian2 synapse weights (which requires slow interpreted mode),
this approach:
1. Runs the fast C++ standalone simulation via run_trial
2. Identifies which KC fired (from spike trains)
3. Stores KC→stimulus associations as "fly memory"
4. On subsequent simulations, post-processes MBON spike rates using learned
   weight multipliers to predict modified behavior

This gives biologically-reasonable learning effects with fast simulation times.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MEMORY_FILE = DATA_DIR / 'fly_memory_fast.json'
MB_NEURONS_FILE = DATA_DIR / 'mushroom_body_neurons.json'


class FastFlyMemory:
    """Fast learning system using post-hoc spike rate modulation."""

    def __init__(self):
        self.comp = pd.read_csv(DATA_DIR / '2025_Completeness_783.csv', index_col=0)
        self.flyid2i = {fid: i for i, fid in enumerate(self.comp.index)}
        self.i2flyid = {i: fid for fid, i in self.flyid2i.items()}

        with open(MB_NEURONS_FILE) as f:
            mb = json.load(f)

        # Build index sets
        kc_ids = mb['kenyon_cells']['KCg'] + mb['kenyon_cells']['KCab']
        self.kc_indices = set(self.flyid2i[fid] for fid in kc_ids if fid in self.flyid2i)

        mbon_ids = []
        for v in mb['mbon'].values():
            mbon_ids.extend(v)
        self.mbon_indices = set(self.flyid2i[fid] for fid in mbon_ids if fid in self.flyid2i)

        pam_ids = []
        for v in mb['dan_pam_reward'].values():
            pam_ids.extend(v)
        self.pam_indices = set(self.flyid2i[fid] for fid in pam_ids if fid in self.flyid2i)

        ppl_ids = []
        for v in mb['dan_ppl_punishment'].values():
            ppl_ids.extend(v)
        self.ppl_indices = set(self.flyid2i[fid] for fid in ppl_ids if fid in self.flyid2i)

        # Load or init memory
        self.memory = self._load_memory()

    def _load_memory(self):
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE) as f:
                return json.load(f)
        return {
            'experiences': [],
            'kc_associations': {},  # kc_index -> {reward_score, punishment_score}
        }

    def save(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def get_active_kc(self, spk_trn):
        """Get KC that fired during simulation."""
        return [idx for idx in spk_trn if idx in self.kc_indices and len(spk_trn[idx]) > 0]

    def get_active_mbon(self, spk_trn):
        """Get MBON activity."""
        return {idx: len(spk_trn[idx]) for idx in spk_trn
                if idx in self.mbon_indices and len(spk_trn[idx]) > 0}

    def apply_punishment(self, active_kc, strength=1.0, label="punishment"):
        """Record punishment for active KC pattern."""
        for kc in active_kc:
            key = str(kc)
            if key not in self.memory['kc_associations']:
                self.memory['kc_associations'][key] = {'reward': 0.0, 'punishment': 0.0}
            self.memory['kc_associations'][key]['punishment'] += strength

        self.memory['experiences'].append({
            'type': 'punishment',
            'label': label,
            'strength': strength,
            'kc_count': len(active_kc),
        })
        self.save()

        return {
            'signal': 'PPL1 多巴胺 (惩罚)',
            'kc_affected': len(active_kc),
            'total_experiences': len(self.memory['experiences']),
        }

    def apply_reward(self, active_kc, strength=1.0, label="reward"):
        """Record reward for active KC pattern."""
        for kc in active_kc:
            key = str(kc)
            if key not in self.memory['kc_associations']:
                self.memory['kc_associations'][key] = {'reward': 0.0, 'punishment': 0.0}
            self.memory['kc_associations'][key]['reward'] += strength

        self.memory['experiences'].append({
            'type': 'reward',
            'label': label,
            'strength': strength,
            'kc_count': len(active_kc),
        })
        self.save()

        return {
            'signal': 'PAM 多巴胺 (奖赏)',
            'kc_affected': len(active_kc),
            'total_experiences': len(self.memory['experiences']),
        }

    def compute_valence(self, active_kc):
        """
        Compute the learned valence for a set of active KC.
        Returns a score: negative = learned avoidance, positive = learned approach.
        Also returns the MBON modulation factor.
        """
        if not active_kc:
            return 0.0, 1.0

        total_reward = 0.0
        total_punishment = 0.0
        n_recognized = 0

        for kc in active_kc:
            key = str(kc)
            if key in self.memory['kc_associations']:
                assoc = self.memory['kc_associations'][key]
                total_reward += assoc['reward']
                total_punishment += assoc['punishment']
                n_recognized += 1

        if n_recognized == 0:
            return 0.0, 1.0  # No memory of this pattern

        # Valence: positive = approach, negative = avoid
        valence = (total_reward - total_punishment) / n_recognized

        # MBON modulation: punishment weakens output, reward strengthens
        # Scale so that 1 punishment → 0.5x, 2 punishments → 0.25x, etc.
        net_score = (total_reward - total_punishment) / n_recognized
        if net_score < 0:
            # Punishment dominant: reduce MBON output
            mbon_factor = max(0.05, 1.0 / (1.0 + abs(net_score) * 2))
        elif net_score > 0:
            # Reward dominant: enhance MBON output
            mbon_factor = min(3.0, 1.0 + net_score * 0.5)
        else:
            mbon_factor = 1.0

        return valence, mbon_factor

    def modulate_results(self, spk_trn, duration_sec):
        """
        Post-process simulation results with learned memory.
        Returns learning info dict.
        """
        active_kc = self.get_active_kc(spk_trn)
        active_mbon = self.get_active_mbon(spk_trn)

        if not active_kc:
            return {
                'active_kc': 0,
                'active_mbon': len(active_mbon),
                'recognized_kc': 0,
                'valence': 0.0,
                'mbon_factor': 1.0,
                'learning_status': 'no_kc_activity',
            }

        valence, mbon_factor = self.compute_valence(active_kc)

        # Count how many KC are "recognized" (have prior associations)
        recognized = sum(1 for kc in active_kc
                        if str(kc) in self.memory['kc_associations'])

        # Describe the effect
        if valence < -0.3:
            status = 'strong_avoidance'
            desc = '果蝇记得这个气味是危险的！强烈回避反应'
        elif valence < 0:
            status = 'mild_avoidance'
            desc = '果蝇对这个气味有些警觉'
        elif valence > 0.3:
            status = 'strong_approach'
            desc = '果蝇记得这个气味是好的！积极趋近'
        elif valence > 0:
            status = 'mild_approach'
            desc = '果蝇对这个气味有些好感'
        else:
            status = 'neutral'
            desc = '果蝇对这个气味没有特别的记忆'

        return {
            'active_kc': len(active_kc),
            'active_mbon': len(active_mbon),
            'recognized_kc': recognized,
            'recognition_pct': round(recognized / len(active_kc) * 100, 1) if active_kc else 0,
            'valence': round(valence, 3),
            'mbon_factor': round(mbon_factor, 3),
            'learning_status': status,
            'description': desc,
            'total_experiences': len(self.memory['experiences']),
        }

    def get_summary(self):
        n_kc_learned = len(self.memory['kc_associations'])
        return {
            'total_experiences': len(self.memory['experiences']),
            'kc_with_associations': n_kc_learned,
            'recent': self.memory['experiences'][-5:],
        }

    def reset(self):
        self.memory = {'experiences': [], 'kc_associations': {}}
        self.save()
        return '果蝇的记忆已清除！（失忆了）🔄'
