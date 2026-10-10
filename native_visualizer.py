"""Record a checkpoint's native decisions and inspect masked placement maps."""
import argparse
import json
from pathlib import Path
import numpy as np

from policy_inspection import checkpoint_config, load_policy
from training_runtime import load_object

CARD_NAMES = ['Wait', 'Knight', 'Archers', 'Minions', 'Arrows', 'Fireball', 'Giant', 'Musketeer', 'Mini PEKKA']


def write_replay(path, frames):
    """Standalone HTML: frame slider, slot probabilities, legal tile heatmaps."""
    payload = json.dumps({'frames': frames, 'names': CARD_NAMES}, allow_nan=False).replace('<', '\\u003c')
    html = '''<!doctype html><meta charset="utf-8"><title>Native policy replay</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;margin:32px}main{max-width:1100px;margin:auto}.maps{display:flex;gap:24px;flex-wrap:wrap}canvas{width:270px;height:480px;image-rendering:pixelated;border:1px solid #475569}select,input{margin:12px}pre{white-space:pre-wrap}label{display:block}</style>
<main><h1>Policy decisions and legal placement probabilities</h1>
<label>Decision <input id="frame" type="range" min="0" value="0"><span id="position"></span></label>
<label>Card slot <select id="card"></select></label><pre id="info"></pre>
<div class="maps"><div>Arena occupancy<canvas id="arena" width="18" height="32"></canvas></div><div>P(tile | card), masked<canvas id="map" width="18" height="32"></canvas></div></div>
<p>Bottom is the learner's side. Dark tiles are illegal. Brightness is relative to the selected card's maximum probability.</p></main>
<script>const data=PAYLOAD,frames=data.frames,slider=document.getElementById('frame'),select=document.getElementById('card');slider.max=frames.length-1;
for(let i=0;i<4;i++){let o=document.createElement('option');o.value=i;select.append(o)}
function draw(){let n=+slider.value,f=frames[n],c=+select.value;document.getElementById('position').textContent=`${n+1}/${frames.length}`;
for(let i=0;i<4;i++)select.options[i].textContent=`Slot ${i+1}: ${data.names[f.hand[i]]}`;
document.getElementById('info').textContent=JSON.stringify({tick:f.tick,seed:f.seed,role:f.role,action:f.action,accepted:f.accepted,slot_probabilities:f.slot_probabilities},null,2);
let a=document.getElementById('arena').getContext('2d'),m=document.getElementById('map').getContext('2d'),max=Math.max(...f.probabilities[c]);
for(let y=0;y<32;y++)for(let x=0;x<18;x++){let k=y*18+x,py=31-y;a.fillStyle=f.own[k]&&f.enemy[k]?'#c084fc':f.own[k]?'#60a5fa':f.enemy[k]?'#fb7185':'#1e293b';a.fillRect(x,py,1,1);let v=max?f.probabilities[c][k]/max:0;m.fillStyle=f.legal[c][k]?`rgb(${Math.round(30+220*v)},${Math.round(45+150*v)},35)`:'#0b1020';m.fillRect(x,py,1,1)}
if(f.action[0]===c+1){m.fillStyle='#ffffff';m.fillRect(f.action[2],31-f.action[1],1,1)}}slider.oninput=draw;select.onchange=draw;draw();</script>'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html.replace('PAYLOAD', payload))
    return path


def record_replay(config, model, steps=64, seed=2_900_001, port=None):
    kwargs = dict(config.get('environment_kwargs', {}))
    kwargs.update(battles=1, seed=seed, decision_ticks=config['decision_ticks'], port=port or config['port'])
    kwargs.pop('opponent_policy', None)
    env = load_object(config['environment'])(**kwargs)
    frames, state, starts = [], None, np.ones(1, dtype=bool)
    try:
        observation = env.reset()
        for _ in range(steps):
            actions, state = model.predict(observation, state=state, episode_start=starts, deterministic=True)
            head = model.policy.action_net
            head.heatmaps()  # Require the spatial policy rather than showing fabricated maps.
            distribution = model.policy.action_dist
            probabilities = np.stack([item.probs.detach().cpu().numpy()[0]
                                      for item in distribution.position_distributions])
            legal = observation['action_mask'][0, 1:].reshape(4, -1).astype(bool)
            probabilities = np.where(legal, probabilities, 0)
            frame = dict(hand=observation['hand'][0].tolist(), action=actions[0].tolist(),
                         own=observation['spatial'][0, 0].reshape(-1).tolist(),
                         enemy=observation['spatial'][0, 1].reshape(-1).tolist(),
                         probabilities=probabilities.tolist(), legal=legal.tolist(),
                         slot_probabilities=distribution.slot_distribution.probs.detach().cpu().numpy()[0].tolist())
            observation, _, dones, infos = env.step(actions)
            frame.update(tick=infos[0]['tick'] - config['decision_ticks'], seed=infos[0]['seed'],
                         role=infos[0]['learner_role'], accepted=infos[0]['deployment_succeeded'])
            frames.append(frame)
            starts = dones
        return frames
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps'))
    parser.add_argument('--port', type=int)
    parser.add_argument('--seed', type=int, default=2_900_001)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('artifacts/policy_replay.html'))
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error('steps must be positive')
    config = checkpoint_config(args.checkpoint, args.config)
    model = load_policy(args.checkpoint, config, args.device)
    frames = record_replay(config, model, args.steps, args.seed, args.port)
    write_replay(args.output, frames)
    args.output.with_suffix('.json').write_text(json.dumps(frames, allow_nan=False))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
