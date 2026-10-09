"""Native deployment audit. Bootstrap the supported emulator first.

Run from the repository root with PYTHONPATH=. using the training interpreter.
Only detached test matches are modified; production engine code is untouched.
"""

import json, subprocess
from pathlib import Path
import frida
from native_engine import Engine,Play
ADB='/Users/jiangzhihao/Library/Android/sdk/platform-tools/adb'
pid=int(subprocess.check_output([ADB,'-s','emulator-5554','shell','pidof','nullsroyale.rel.free'],text=True).strip())
port=subprocess.check_output([ADB,'-s','emulator-5554','forward','tcp:0','tcp:27042'],text=True).strip()
device=frida.get_device_manager().add_remote_device('127.0.0.1:'+port)
session=device.attach(pid)
script=session.create_script('''
rpc.exports = {
  weaken(managerAddress, owner, xs) {
    const world=ptr(managerAddress).add(0xa8).readPointer();
    const om=world.add(0xe0).readPointer().add(0x10).readPointer();
    const count=om.add(0x14).readS32();
    if (count < 6 || count > 100000) throw Error('invalid object vector');
    const items=om.add(8).readPointer(), changed=[];
    for (let i=0;i<count;i++) {
      const o=items.add(i*8).readPointer();
      if(o.isNull() || o.add(0xac).readS32()!==-1 || o.add(0x78).readS32()!==owner) continue;
      const x=o.add(0x7c).readS32(),y=o.add(0x80).readS32();
      if(!xs.includes(x) || y !== (owner===0?6500:25500)) continue;
      const components=o.add(0x18).readPointer();
      const hp=components.add(16).readPointer();
      if(!hp.add(8).readPointer().equals(o)) throw Error('invalid HP component backlink');
      changed.push({owner,x,y,hp:hp.add(0x10).readS32()});
      hp.add(0x10).writeS32(1);
    }
    if(changed.length!==xs.length) throw Error('missing tower');
    return changed;
  }
};
''')
script.load()
engine=Engine(timeout=10)

def fixture(owner, lanes):
    global fixture_initial
    s=engine.reset(3)
    fixture_initial=s
    xs=[3500 if (lane==0)==(owner==0) else 14500 for lane in lanes]
    # Only weaken detached test-fixture towers; a normal spell triggers native death.
    script.exports_sync.weaken(engine.request('status')['manager'],1-owner,xs)
    for x in xs:
        spells=[c for c in s.players[owner].hand if c.card in (28000000,28000001)]
        if not spells: raise RuntimeError('fixture requires a damage spell')
        c=spells[0]
        if s.players[owner].elixir < c.cost*10000:
            s=engine.step((),ticks=200).state
        s=engine.step((Play(owner,c.slot,x,25500 if owner==0 else 6500),),ticks=200).state
    alive=[e.x for e in s.entities if e.card==-1 and e.owner==1-owner and e.hp>0]
    if any(x in alive for x in xs): raise RuntimeError('tower not destroyed')
    if s.finalized: raise RuntimeError('fixture finalized')
    return s


from native_engine.native_vec_env import NativeBatchVecEnv
probe_env=NativeBatchVecEnv(battles=1)
report={'rules':probe_env.deployment_rules_version,'states':{},'mismatches':[],'fixture':'detached tower HP reduced to 1; normal spell performs native destruction'}
try:
    for owner in (0,1):
        for lanes in ((),(0,),(1,),(0,1)):
            key=str(owner)+':'+(','.join(map(str,lanes)) or 'base')
            tested=allowed=0
            for y in range(32):
                for x in range(18):
                    if lanes:
                        s=fixture(owner,lanes); initial=fixture_initial
                    else:
                        s=engine.reset(3); initial=s
                    c=next(c for c in s.players[owner].hand if c.card in (26000000,26000003,26000014,26000018))
                    if s.players[owner].elixir<c.cost*10000: s=engine.step((),ticks=200).state
                    probe_env._begin_episode(probe_env.slots[0],initial,3)
                    mask=probe_env._action_mask(s,owner,probe_env.slots[0].initial_towers)
                    hand_index=next(i for i,card in enumerate(s.players[owner].hand) if card.slot==c.slot)
                    intended=bool(mask[hand_index+1,y,x]); allowed+=int(intended)
                    px,py=x*1000+500,y*1000+500
                    if owner==1: px,py=18000-px,32000-py
                    t=engine.step((Play(owner,c.slot,px,py),),ticks=2)
                    objects=[e for e in t.state.entities if e.owner==owner and e.card==c.card]
                    actual=bool(t.executed[0] and any(abs(e.x-px)<20 and (abs(e.y-py)<20 or (py==31500 and e.y==31000)) for e in objects))
                    tested+=1
                    if actual!=intended:
                        report['mismatches'].append({'state':key,'tile':[y,x],'mask':intended,'executed':t.executed[0],'spawn':[[e.x,e.y] for e in objects]})
            report['states'][key]={'tested':tested,'allowed':allowed}
            print(key,report['states'][key],'mismatches',len(report['mismatches']),flush=True)
            Path('artifacts/deployment_validation/native_validation.json').write_text(json.dumps(report,indent=2))
    if report['mismatches']: raise RuntimeError('native deployment mask mismatches')
finally:
    probe_env.close();engine.close();script.unload();session.detach()
    subprocess.run([ADB,'-s','emulator-5554','forward','--remove','tcp:'+port],check=True)
