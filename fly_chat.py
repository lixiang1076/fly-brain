#!/usr/bin/env python3
"""
fly_chat.py — Interactive command-line interface to "chat" with a fruit fly brain.

Translates natural language inputs to neural stimulations, runs the whole-brain
LIF simulation, and translates the results back to behavioral descriptions.

Usage:
    python fly_chat.py
"""

import json
import sys
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

BEHAVIOR_DESCRIPTIONS = {
    'forward_walk': {
        'strong':   '🦶 果蝇大步向前走！六条腿协调运动，步态稳健有力。',
        'moderate': '🦶 果蝇在缓缓向前移动，像在散步。',
        'weak':     '🦶 果蝇的腿微微动了一下，似乎想走但不太确定。',
    },
    'backward_walk': {
        'strong':   '🔙 果蝇猛地向后退！这是典型的惊吓反应——"月球漫步"模式！',
        'moderate': '🔙 果蝇在小心翼翼地向后退。',
        'weak':     '🔙 果蝇似乎有一点想后退的倾向。',
    },
    'turning': {
        'strong':   '🔄 果蝇剧烈转向！它改变了前进方向。',
        'moderate': '🔄 果蝇在调整方向，慢慢转弯。',
        'weak':     '🔄 果蝇身体微微偏转。',
    },
    'escape': {
        'strong':   '🚀 紧急起飞！巨纤维回路全力激活——果蝇弹射升空，这是它最快的逃命反应！',
        'moderate': '🚀 果蝇的逃跑回路被激活了，翅膀在准备起飞。',
        'weak':     '🚀 果蝇有些警觉，逃跑回路轻微活动。',
    },
    'feeding': {
        'strong':   '👅 果蝇伸出了口器（proboscis）！它在积极进食，看起来很享受。',
        'moderate': '👅 果蝇的口器在动，似乎在尝味道。',
        'weak':     '👅 口器运动神经元轻微活动，果蝇对食物有点兴趣。',
    },
    'grooming': {
        'strong':   '🧹 果蝇在认真梳理触角！前腿举起来反复擦拭。',
        'moderate': '🧹 果蝇开始梳理自己了，触角区域。',
        'weak':     '🧹 果蝇有轻微的梳理倾向。',
    },
}

STIMULUS_MATCH_RULES = [
    # (pattern_list, stim_key, optional_extra_stim, description)
    (['甜', '糖', 'sugar', 'sweet', '好吃', '蜂蜜', '食物', '吃东西', '喂食'], 
     'taste_sweet', None, '给果蝇尝甜的'),
    (['苦', 'bitter', '毒', '难吃', '有毒'],
     'taste_bitter', None, '给果蝇尝苦的'),
    (['甜.*苦', '苦.*甜', '酸甜苦辣'],
     'taste_sweet', 'taste_bitter', '同时给甜味和苦味'),
    (['走', '走路', '前进', 'walk', 'forward', '散步', '移动', '跑'],
     'walk_forward', None, '让果蝇走路'),
    (['后退', '退', 'backward', '月球', 'moonwalk'],
     'walk_backward', None, '让果蝇后退'),
    (['逃', 'escape', '危险', '天敌', '拍', '打', '快跑'],
     'escape', None, '触发逃跑反应'),
    (['转', 'turn', '拐', '方向'],
     'turn', None, '让果蝇转向'),
    (['梳理', '清洁', 'groom', '洗', '痒', '抓'],
     'groom', None, '让果蝇梳理触角'),
    (['看', '视觉', '眼', '阴影', '黑影', '碰撞', 'vision', 'looming'],
     'vision_looming', None, '视觉威胁检测'),
    (['听', '声', 'hear', 'sound', '音乐', '风', '气流'],
     'hearing', None, '给果蝇听觉刺激'),
    (['闻', '嗅', 'smell', '气味', '臭', '霉', '腐', '难闻'],
     'smell_danger', None, '给果蝇闻到危险气味'),
    # Combo: 走路+看到危险
    (['走.*看', '走.*碰', '走.*危', '边走边'],
     'walk_forward', 'vision_looming', '走路时遇到视觉威胁'),
    # Combo: 走路+闻到东西
    (['走.*闻', '走.*味', '边走.*嗅'],
     'walk_forward', 'smell_danger', '走路时闻到危险气味'),
]


def match_stimulus(text):
    """Match user text to stimulus keys using keyword rules."""
    text = text.lower().strip()

    # Check combo patterns first (longer patterns take priority)
    for patterns, stim1, stim2, desc in STIMULUS_MATCH_RULES:
        if stim2 is not None:  # combo pattern
            for pat in patterns:
                if re.search(pat, text):
                    return [stim1, stim2], desc

    # Check single patterns
    for patterns, stim1, stim2, desc in STIMULUS_MATCH_RULES:
        if stim2 is None:  # single pattern
            for pat in patterns:
                if re.search(pat, text):
                    return [stim1], desc

    return None, None


def parse_silence(text):
    """Check if user wants to silence something."""
    silence_patterns = {
        '关.*视觉|没有眼|闭眼|瞎|blind': 'vision_looming',
        '关.*听觉|聋|deaf|没有耳': 'hearing',
        '关.*嗅觉|没有鼻|闻不到': 'smell_danger',
        '关.*味觉|尝不到': 'taste_sweet',
    }
    silenced = []
    for pat, key in silence_patterns.items():
        if re.search(pat, text.lower()):
            silenced.append(key)
    return silenced if silenced else None


def format_result(result):
    """Format simulation result into a human-readable narrative."""
    lines = []
    lines.append('')
    lines.append('=' * 50)
    lines.append(f'🪰  果蝇大脑仿真完成！')
    lines.append(f'    仿真时长: {result["duration_sec"]}秒 (用时 {result["wall_time_sec"]}秒)')
    lines.append(f'    活跃神经元: {result["active_neurons"]}/{result["total_neurons"]}')
    lines.append(f'    总脉冲数: {result["total_spikes"]}')
    lines.append('=' * 50)

    # Stimulation info
    lines.append('')
    lines.append('📡 刺激:')
    for s in result['stimulation']:
        lines.append(f'    • {s}')

    # Behavior narrative
    behaviors = result.get('predicted_behaviors', [])
    if behaviors:
        lines.append('')
        lines.append('🧠 果蝇的反应:')
        lines.append('')
        for beh in behaviors:
            bname = beh['behavior']
            intensity = beh['intensity']
            desc_map = BEHAVIOR_DESCRIPTIONS.get(bname, {})
            desc = desc_map.get(intensity, f'{bname}: {intensity} (score={beh["score"]})')
            lines.append(f'    {desc}')
    else:
        lines.append('')
        lines.append('😴 果蝇没有明显的行为反应... 可能刺激太弱了。')

    # Output neuron detail (compact)
    lines.append('')
    lines.append('📊 关键输出神经元活动:')
    active_outputs = {k: v for k, v in result['output_neuron_activity'].items() if v['rate_hz'] > 0}
    if active_outputs:
        for name, act in sorted(active_outputs.items(), key=lambda x: -x[1]['rate_hz']):
            bar_len = min(int(act['rate_hz'] / 5), 30)
            bar = '█' * bar_len
            lines.append(f'    {name:20s} {act["rate_hz"]:6.1f} Hz  {bar}  ({act["function"]})')
    else:
        lines.append('    (无输出神经元活动)')

    lines.append('')
    return '\n'.join(lines)


def print_help():
    print("""
╔══════════════════════════════════════════════════════╗
║      🪰  果蝇大脑对话系统 (带学习功能) 🧠           ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  感觉刺激:                                           ║
║  🍬 味觉:  "给果蝇尝甜的" / "让它尝尝苦的"         ║
║  🦶 运动:  "让果蝇走路" / "让果蝇后退"             ║
║  🚀 逃跑:  "拍果蝇" / "有危险"                     ║
║  🧹 梳理:  "让果蝇梳理触角"                         ║
║  👁  视觉:  "让果蝇看到黑影"                        ║
║  👂 听觉:  "给果蝇听声音"                           ║
║  👃 嗅觉:  "让果蝇闻到臭味"                         ║
║                                                      ║
║  🧠 学习 (多巴胺调节):                              ║
║  ⚡ "电击"    - 惩罚! 果蝇学会回避上次的刺激        ║
║  🍰 "奖励"    - 奖赏! 果蝇学会喜欢上次的刺激       ║
║  📋 "记忆"    - 查看果蝇学到了什么                   ║
║  🔄 "失忆"    - 清除所有记忆                         ║
║                                                      ║
║  组合: "让果蝇边走边看到危险"                        ║
║  关闭: "关掉视觉" + 其他指令                         ║
║                                                      ║
║  命令: help / list / quit                            ║
╚══════════════════════════════════════════════════════╝
""")


def print_stimulus_list():
    with open(BASE_DIR / 'neuron_atlas.json') as f:
        atlas = json.load(f)
    print('\n📋 可用刺激列表:\n')
    for key, info in atlas['stimuli'].items():
        n = len(info.get('neuron_ids', []))
        if n == 0 and 'neuron_ids_groups' in info:
            n = sum(len(v) for v in info['neuron_ids_groups'].values())
        print(f'  {key:20s}  {info["label"]}')
        print(f'  {"":20s}  {info["description"]}')
        print(f'  {"":20s}  默认频率: {info["default_freq_hz"]}Hz | 神经元数: {n}')
        print(f'  {"":20s}  预期行为: {info["behavior_expected"]}')
        print()


def handle_learning_command(text, last_result):
    """Handle learning-related commands (reward/punishment/memory/reset)."""
    from dopamine_learning import FlyMemory

    fly_mem = FlyMemory()
    text_lower = text.lower()

    # Memory check
    if any(kw in text_lower for kw in ['记忆', 'memory', '学到', '回忆']):
        summary = fly_mem.get_memory_summary()
        print(f'\n🧠 果蝇的记忆:')
        print(f'   总经历: {summary["total_experiences"]} 次')
        print(f'   已修改突触: {summary["modified_synapses"]} 条')
        if summary['recent_experiences']:
            print(f'   最近经历:')
            for exp in summary['recent_experiences']:
                icon = '⚡' if exp['signal_type'] == 'punishment' else '🍰'
                print(f'     {icon} {exp["label"]} ({exp["signal_type"]}, '
                      f'影响 {exp["active_kc_count"]} 个 KC)')
        else:
            print(f'   这只果蝇还没有任何记忆（新生儿）')
        return True

    # Memory reset
    if any(kw in text_lower for kw in ['失忆', '清除记忆', 'amnesia', 'reset memory', '重置']):
        result = fly_mem.reset_memory()
        print(f'\n🔄 {result["message"]}')
        return True

    # Punishment
    if any(kw in text_lower for kw in ['电击', '惩罚', 'shock', 'punish', '打它', '电它']):
        if not last_result or 'mushroom_body' not in last_result:
            print('\n⚠️ 需要先给果蝇一个刺激（比如"给果蝇尝甜的"），'
                  '然后才能电击它来让它学习。')
            return True

        active_kc = last_result['mushroom_body'].get('active_kc_indices', [])
        if not active_kc:
            print('\n⚠️ 上次仿真中没有 KC 被激活，没什么可学的。')
            return True

        result = fly_mem.apply_punishment(active_kc, strength=0.3,
                                          label=f'惩罚-{last_result["stimulation"][0]}')
        print(f'\n⚡ 电击！PPL1 多巴胺神经元激活！')
        print(f'   修改了 {result["synapses_modified"]} 条 KC→MBON 突触')
        print(f'   {result["active_kc"]} 个活跃 KC 的输出被削弱')
        print(f'   果蝇学会了：上次的刺激 = 危险！ ⚡')
        print(f'   (总经历: {result["total_experiences"]} 次)')
        print(f'\n   💡 现在再给它同样的刺激试试，看它的反应有没有变化！')
        return True

    # Reward
    if any(kw in text_lower for kw in ['奖励', '奖赏', 'reward', '奖', '好的']):
        if not last_result or 'mushroom_body' not in last_result:
            print('\n⚠️ 需要先给果蝇一个刺激，然后才能奖励它。')
            return True

        active_kc = last_result['mushroom_body'].get('active_kc_indices', [])
        if not active_kc:
            print('\n⚠️ 上次仿真中没有 KC 被激活，没什么可学的。')
            return True

        result = fly_mem.apply_reward(active_kc, strength=1.5,
                                       label=f'奖赏-{last_result["stimulation"][0]}')
        print(f'\n🍰 奖赏！PAM 多巴胺神经元激活！')
        print(f'   增强了 {result["synapses_modified"]} 条 KC→MBON 突触')
        print(f'   {result["active_kc"]} 个活跃 KC 的输出被增强')
        print(f'   果蝇学会了：上次的刺激 = 好东西！ 🍬')
        print(f'   (总经历: {result["total_experiences"]} 次)')
        return True

    return False


def main():
    print_help()
    last_result = None  # Store last simulation result for learning

    while True:
        try:
            text = input('\n🪰 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n\n👋 再见！果蝇回去休息了。')
            break

        if not text:
            continue

        if text.lower() in ('quit', 'exit', 'q', '退出', '再见'):
            print('\n👋 再见！果蝇回去休息了。')
            break

        if text.lower() in ('help', 'h', '帮助', '?', '？'):
            print_help()
            continue

        if text.lower() in ('list', 'ls', '列表', '刺激'):
            print_stimulus_list()
            continue

        # Check learning commands first
        try:
            if handle_learning_command(text, last_result):
                continue
        except Exception as e:
            print(f'\n❌ 学习操作出错: {e}')
            continue

        # Match stimulus
        stim_keys, desc = match_stimulus(text)
        silence_keys = parse_silence(text)

        if not stim_keys and not silence_keys:
            print('\n🤔 我不太理解你想对果蝇做什么...')
            print('   试试 "给果蝇尝甜的"、"让果蝇走路"、"拍果蝇" 等。')
            print('   学习命令: "电击"、"奖励"、"记忆"、"失忆"')
            print('   输入 help 查看所有可用命令。')
            continue

        if not stim_keys and silence_keys:
            print('\n🤔 你只指定了关闭某些感觉，但没说让果蝇做什么。')
            print('   试试 "关掉视觉让果蝇走路"')
            continue

        # Describe what we're about to do
        print(f'\n🔬 理解: {desc}')
        if silence_keys:
            print(f'   同时关闭: {", ".join(silence_keys)}')
        print(f'   开始仿真 13.8万个神经元...')

        # Run simulation
        try:
            from chat_with_fly import run_simulation
            result = run_simulation(
                stim_keys=stim_keys,
                silence_keys=silence_keys,
                duration_sec=0.1,
                use_memory=True,
            )
            last_result = result  # Save for learning
            print(format_result(result))

            # Show KC info if available
            if 'mushroom_body' in result:
                mb = result['mushroom_body']
                print(f'    🧠 蘑菇体: {mb["active_kc_count"]}/{mb["total_kc"]} KC 激活, '
                      f'{mb["active_mbon_count"]}/{mb["total_mbon"]} MBON 活跃')
                if mb['active_kc_count'] > 0:
                    print(f'    💡 输入 "电击" 或 "奖励" 来让果蝇学习这次经历')
                print()

        except Exception as e:
            print(f'\n❌ 仿真出错: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
