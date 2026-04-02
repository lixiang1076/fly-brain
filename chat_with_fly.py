#!/usr/bin/env python3
"""
chat_with_fly.py — Simulation wrapper for the Drosophila whole-brain model.

Accepts neuron activation/silencing parameters, runs the Brian2 LIF simulation,
and returns structured results (spike rates by output neuron group, behavior
predictions, etc.) as JSON to stdout.

Usage:
    python chat_with_fly.py --stim taste_sweet --freq 200 --duration 0.1
    python chat_with_fly.py --stim walk_forward,taste_sweet --freq 100,200 --duration 0.2
    python chat_with_fly.py --stim walk_forward --silence vision_looming --duration 0.1
    python chat_with_fly.py --neuron-ids 720575940627652358,720575940635872101 --freq 100 --duration 0.1
"""

import argparse
import json
import sys
import os
import warnings
import tempfile
from pathlib import Path
from time import time

warnings.filterwarnings('ignore', category=UserWarning)
os.environ['PYTHONUNBUFFERED'] = '1'

# Paths
BASE_DIR = Path(__file__).resolve().parent
ATLAS_PATH = BASE_DIR / 'neuron_atlas.json'
DATA_DIR = BASE_DIR / 'data'
CODE_DIR = BASE_DIR / 'code' / 'paper-phil-drosophila'

sys.path.insert(0, str(CODE_DIR))


def load_atlas():
    with open(ATLAS_PATH) as f:
        return json.load(f)


def get_neuron_ids_for_stimulus(atlas, stim_key):
    """Get flat list of neuron IDs for a stimulus key."""
    stim = atlas['stimuli'].get(stim_key)
    if not stim:
        raise ValueError(f"Unknown stimulus: {stim_key}. Available: {list(atlas['stimuli'].keys())}")
    if 'neuron_ids' in stim:
        return stim['neuron_ids']
    elif 'neuron_ids_groups' in stim:
        ids = []
        for group_ids in stim['neuron_ids_groups'].values():
            ids.extend(group_ids)
        return ids
    return []


def build_output_neuron_index(atlas, df_comp):
    """Build mapping from flywire_id -> output neuron name."""
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
    output_map = {}
    for name, info in atlas['output_neurons'].items():
        fid = info['id']
        if fid in flyid2i:
            output_map[fid] = {'name': name, 'function': info['function'], 'index': flyid2i[fid]}
    return output_map


def analyze_results(spk_trn, output_map, duration_sec, atlas, df_comp):
    """Analyze spike train results and produce a structured summary."""
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
    i2flyid = {i: fid for fid, i in flyid2i.items()}

    # Compute spike rates for output neurons
    output_activity = {}
    for fid, info in output_map.items():
        idx = info['index']
        if idx in spk_trn:
            n_spikes = len(spk_trn[idx])
            rate = n_spikes / duration_sec
        else:
            n_spikes = 0
            rate = 0.0
        output_activity[info['name']] = {
            'function': info['function'],
            'spikes': n_spikes,
            'rate_hz': round(rate, 1),
        }

    # Summary stats
    total_spikes = sum(len(v) for v in spk_trn.values())
    active_neurons = len(spk_trn)

    # Behavior prediction based on output neuron activity
    behaviors = []
    behavior_scores = {
        'forward_walk': 0,
        'backward_walk': 0,
        'turning': 0,
        'escape': 0,
        'feeding': 0,
        'grooming': 0,
        'mating_acceptance': 0,
        'oviposition': 0,
        'mating_rejection': 0,
    }

    for name, act in output_activity.items():
        r = act['rate_hz']
        if r > 0:
            if 'P9_oDN1' in name:
                behavior_scores['forward_walk'] += r
            elif 'MDN' in name:
                behavior_scores['backward_walk'] += r
            elif 'DNa0' in name:
                behavior_scores['turning'] += r
            elif 'Giant_Fiber' in name:
                behavior_scores['escape'] += r
            elif 'MN9' in name:
                behavior_scores['feeding'] += r
            elif 'aDN1' in name:
                behavior_scores['grooming'] += r
            elif 'vpoDN' in name:
                behavior_scores['mating_acceptance'] += r
            elif 'oviDN' in name:
                behavior_scores['oviposition'] += r
            elif 'DNp13' in name:
                behavior_scores['mating_rejection'] += r

    # Sort behaviors by score
    for beh, score in sorted(behavior_scores.items(), key=lambda x: -x[1]):
        if score > 0:
            intensity = 'strong' if score > 50 else 'moderate' if score > 10 else 'weak'
            behaviors.append({'behavior': beh, 'score': round(score, 1), 'intensity': intensity})

    return {
        'total_spikes': total_spikes,
        'active_neurons': active_neurons,
        'total_neurons': len(df_comp),
        'output_neuron_activity': output_activity,
        'predicted_behaviors': behaviors,
    }


def run_simulation(stim_keys=None, silence_keys=None, neuron_ids=None,
                   freq_hz=None, duration_sec=0.1, use_memory=True):
    """
    Run the fly brain simulation with optional learned memory.

    Args:
        stim_keys: list of stimulus keys from atlas (e.g., ['taste_sweet'])
        silence_keys: list of stimulus keys to silence
        neuron_ids: explicit list of neuron IDs to stimulate (overrides stim_keys)
        freq_hz: list of frequencies matching stim_keys, or single value
        duration_sec: simulation duration in seconds
        use_memory: if True, apply learned weight modifications from fly_memory.json

    Returns:
        dict with simulation results + learning info
    """
    import pandas as pd
    from brian2 import ms, Hz, mV

    atlas = load_atlas()

    path_comp = str(DATA_DIR / '2025_Completeness_783.csv')
    path_con = str(DATA_DIR / '2025_Connectivity_783.parquet')

    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    # Build excitation list
    exc_ids = []
    exc2_ids = []
    stim_desc = []

    if neuron_ids:
        exc_ids = neuron_ids
        stim_desc.append(f"{len(neuron_ids)} custom neurons")
        if freq_hz is None:
            freq_hz = [150]
    elif stim_keys:
        if freq_hz is None:
            freq_hz = []
        # Normalize freq_hz to list
        if isinstance(freq_hz, (int, float)):
            freq_hz = [freq_hz]

        for i, key in enumerate(stim_keys):
            ids = get_neuron_ids_for_stimulus(atlas, key)
            f = freq_hz[i] if i < len(freq_hz) else atlas['stimuli'][key]['default_freq_hz']
            stim_info = atlas['stimuli'][key]
            stim_desc.append(f"{stim_info['label']} @ {f}Hz ({len(ids)} neurons)")

            if i == 0:
                exc_ids.extend(ids)
                primary_freq = f
            else:
                exc2_ids.extend(ids)
                secondary_freq = f
    else:
        raise ValueError("Must specify either stim_keys or neuron_ids")

    # Build silence list
    slnc_ids = []
    if silence_keys:
        for key in silence_keys:
            ids = get_neuron_ids_for_stimulus(atlas, key)
            slnc_ids.extend(ids)
            stim_info = atlas['stimuli'][key]
            stim_desc.append(f"SILENCED: {stim_info['label']} ({len(ids)} neurons)")

    # Convert to indices
    exc_indices = [flyid2i[fid] for fid in exc_ids if fid in flyid2i]
    exc2_indices = [flyid2i[fid] for fid in exc2_ids if fid in flyid2i]
    slnc_indices = [flyid2i[fid] for fid in slnc_ids if fid in flyid2i]

    # Set up params
    from model import default_params, run_trial, create_model, poi, silence, get_spk_trn
    from brian2 import Network
    params = dict(default_params)
    params['t_run'] = duration_sec * 1000 * ms

    if freq_hz:
        params['r_poi'] = freq_hz[0] * Hz
    if len(freq_hz) > 1:
        params['r_poi2'] = freq_hz[1] * Hz

    # Check if we have learned weights to apply
    memory_info = None
    has_learned_weights = False
    if use_memory:
        try:
            from dopamine_learning import FlyMemory, apply_learned_weights
            fly_mem = FlyMemory()
            mods = fly_mem.get_weight_multipliers()
            has_learned_weights = len(mods) > 0
        except Exception:
            pass

    # Run simulation
    t_start = time()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['BRIAN2_DEVICE_DIRECTORY'] = tmpdir

        if has_learned_weights:
            # Manual model creation so we can modify weights before running
            neu, syn, spk_mon = create_model(path_comp, path_con, params)
            poi_inp, neu = poi(neu, exc_indices, exc2_indices, params)
            syn = silence(slnc_indices, syn)

            n_modified = apply_learned_weights(syn, fly_mem)
            memory_info = {
                'learned_weights_applied': n_modified,
                'total_experiences': fly_mem.memory['total_experiences'],
            }
            stim_desc.append(f"🧠 已应用 {n_modified} 条学习记忆")

            net = Network(neu, syn, spk_mon, *poi_inp)
            net.run(duration=params['t_run'])
            spk_trn = get_spk_trn(spk_mon)
        else:
            # Fast path: use run_trial (uses Brian2 default which is fine for short sims)
            spk_trn = run_trial(
                exc=exc_indices,
                exc2=exc2_indices,
                slnc=slnc_indices,
                path_comp=path_comp,
                path_con=path_con,
                params=params,
            )

    elapsed = time() - t_start

    # Analyze
    output_map = build_output_neuron_index(atlas, df_comp)
    analysis = analyze_results(spk_trn, output_map, duration_sec, atlas, df_comp)

    # Extract KC activity for potential learning
    kc_info = None
    if use_memory:
        try:
            from dopamine_learning import FlyMemory
            fly_mem = FlyMemory()
            active_kc = fly_mem.get_active_kc(spk_trn)
            active_mbon = fly_mem.get_active_mbon(spk_trn)
            kc_info = {
                'active_kc_count': len(active_kc),
                'active_mbon_count': len(active_mbon),
                'total_kc': len(fly_mem.kc_indices),
                'total_mbon': len(fly_mem.mbon_indices),
                'active_kc_indices': active_kc,  # Store for learning
            }
        except Exception:
            pass

    result = {
        'status': 'success',
        'stimulation': stim_desc,
        'duration_sec': duration_sec,
        'wall_time_sec': round(elapsed, 2),
        **analysis,
    }

    if memory_info:
        result['memory'] = memory_info
    if kc_info:
        result['mushroom_body'] = kc_info

    return result


def main():
    parser = argparse.ArgumentParser(description='Chat with the fly brain')
    parser.add_argument('--stim', type=str, default=None,
                        help='Comma-separated stimulus keys (e.g., taste_sweet,walk_forward)')
    parser.add_argument('--silence', type=str, default=None,
                        help='Comma-separated stimulus keys to silence')
    parser.add_argument('--neuron-ids', type=str, default=None,
                        help='Comma-separated neuron IDs to stimulate directly')
    parser.add_argument('--freq', type=str, default=None,
                        help='Comma-separated frequencies in Hz (matching --stim order)')
    parser.add_argument('--duration', type=float, default=0.1,
                        help='Simulation duration in seconds (default: 0.1)')
    parser.add_argument('--pretty', action='store_true',
                        help='Pretty-print JSON output')

    args = parser.parse_args()

    stim_keys = args.stim.split(',') if args.stim else None
    silence_keys = args.silence.split(',') if args.silence else None
    neuron_ids = [int(x) for x in args.neuron_ids.split(',')] if args.neuron_ids else None
    freq_hz = [float(x) for x in args.freq.split(',')] if args.freq else None

    result = run_simulation(
        stim_keys=stim_keys,
        silence_keys=silence_keys,
        neuron_ids=neuron_ids,
        freq_hz=freq_hz,
        duration_sec=args.duration,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))


if __name__ == '__main__':
    main()
