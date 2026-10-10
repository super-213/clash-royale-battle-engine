"""Draw method figures for paper.md; no experimental measurements are plotted.

Run with a Python environment containing matplotlib and numpy.
Exports editable vector SVGs and 220-dpi PNGs into ./photo.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

OUT = Path(__file__).resolve().parent / "photo"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "svg.fonttype": "path", "savefig.facecolor": "white",
    "axes.unicode_minus": False,
})
C = {"blue": "#2563a6", "teal": "#147d73", "orange": "#b66a15",
     "purple": "#7950a0", "ink": "#233247", "muted": "#617083",
     "red": "#b3444b", "line": "#79889a", "grey": "#f2f5f8"}


def canvas(w, h, title, subtitle):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=.02, right=.98, bottom=.025, top=.97)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.025, .965, title, fontsize=19, weight="bold", color=C["ink"], va="top")
    ax.text(.025, .915, subtitle, fontsize=11, color=C["muted"], va="top")
    return fig, ax


def box(ax, x, y, w, h, title, detail="", color="blue", fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                              ec=C[color], fc="white", lw=1.5, zorder=3))
    if detail:
        ax.text(x+w/2, y+h*.68, title, ha="center", va="center", fontsize=fs,
                weight="bold", color=C[color], zorder=4)
        ax.text(x+w/2, y+h*.30, detail, ha="center", va="center", fontsize=fs-1,
                color=C["ink"], linespacing=1.4, zorder=4)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=fs,
                color=C[color], weight="bold", zorder=4)


def arrow(ax, points, color="line", dashed=False, label=None):
    style = "--" if dashed else "-"
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=C[color], lw=1.5,
                linestyle=style, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=13, lw=1.5, color=C[color],
                                linestyle=style, zorder=2))
    if label:
        a, b = points[0], points[1]
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.014, label, ha="center",
                fontsize=9, color=C[color], bbox={"fc":"white", "ec":"none", "pad":1}, zorder=5)


def save(fig, name):
    for ext in ("svg", "png"):
        output = OUT / f"{name}.{ext}"
        fig.savefig(output, dpi=220)
        if ext == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


def system():
    fig, ax = canvas(14, 7.5, "01  Native interaction and learning pipeline",
                     "One resident engine process, one batch connection, up to 16 independent battles")
    specs = [(.04,"Native battle engine","State + public events"),
             (.275,"Batch / State.decode","Receipts + structured state"),
             (.51,"Observation adapter","Egocentric tensors + masks"),
             (.745,"Recurrent policy","Action + value + memory")]
    for x,t,d in specs: box(ax,x,.64,.20,.14,t,d,fs=11)
    for x in (.24,.475,.71): arrow(ax,[(x,.71),(x+.035,.71)])
    box(ax,.745,.34,.20,.13,"Play command","[slot, y, x] -> native x/y",color="teal",fs=11)
    arrow(ax,[(.845,.64),(.845,.47)],color="teal")
    arrow(ax,[(.745,.405),(.14,.405),(.14,.64)],color="teal",label="execute next tick; advance 10 ticks (0.5 s)")
    box(ax,.04,.13,.23,.13,"Separate training labels","Entity-ID targets + executed action",color="orange",fs=10)
    box(ax,.36,.13,.24,.13,"Recurrent rollout buffer","Observations / rewards / LSTM states",color="orange",fs=10)
    box(ax,.70,.13,.245,.13,"Joint optimization","PPO + 0.1 x auxiliary loss",color="purple",fs=11)
    arrow(ax,[(.14,.64),(.14,.26)],color="orange",dashed=True)
    arrow(ax,[(.27,.195),(.36,.195)],color="orange",dashed=True)
    arrow(ax,[(.61,.64),(.61,.31),(.48,.31),(.48,.26)],color="orange",dashed=True)
    arrow(ax,[(.60,.195),(.70,.195)],color="orange",dashed=True)
    arrow(ax,[(.945,.195),(.978,.195),(.978,.80),(.845,.80),(.845,.78)],color="purple",dashed=True)
    ax.text(.04,.04,"Solid: interaction and inference     Dashed: training-only data or parameter updates",color=C["muted"],fontsize=11)
    save(fig,"fig01_system_pipeline")


def architecture():
    fig, ax = canvas(16, 10, "02  Spatial, relational and recurrent policy architecture",
                     "Attention context is stated inside each block; complete card and spatial tokens bypass summary fusion")
    rows=[(.77,"Spatial  16 x 32 x 18","Residual CNN + coordinates","576 x 64 spatial tokens","blue"),
          (.64,"Entities  128 x 24","2-layer entity Transformer","128 x 64; masked pooling","blue"),
          (.51,"History  32 x 8","History -> entities attention","Event LSTM -> 64","teal"),
          (.38,"Public belief  20","Belief -> entities + history","1 x 64 contextual token","teal"),
          (.25,"Hand  5 card identities","Cards -> entities + history + belief","5 x 64 contextual tokens","teal"),
          (.12,"Scalars 12 / temporal 20","Separate MLP encoders","32 + 32 dimensions","blue")]
    for y,inp,t,d,c in rows:
        box(ax,.025,y,.19,.09,inp,color=c,fs=11)
        box(ax,.25,y,.28,.09,t,d,color=c,fs=11 if y!=.25 else 10)
        arrow(ax,[(.215,y+.045),(.25,y+.045)],color=c)
        arrow(ax,[(.53,y+.045),(.565,y+.045),(.565,.49),(.60,.49)],color="line")
    box(ax,.60,.40,.155,.18,"Summary fusion","384 -> 256\nMLP + LayerNorm",fs=12)
    box(ax,.795,.65,.17,.10,"Actor LSTM","256 -> MLP 256",color="purple",fs=11)
    box(ax,.795,.45,.17,.10,"Critic LSTM","256 -> value head",color="purple",fs=11)
    arrow(ax,[(.755,.53),(.775,.53),(.775,.70),(.795,.70)],color="purple")
    arrow(ax,[(.755,.47),(.775,.47),(.775,.50),(.795,.50)],color="purple")
    box(ax,.775,.235,.19,.13,"Spatial action head","5 slot logits\n4 x 576 position logits",color="teal",fs=11)
    arrow(ax,[(.88,.65),(.88,.59),(.99,.59),(.99,.30),(.965,.30)],color="purple")
    # Route full spatial tokens outside the recurrent branch, directly into the action head.
    arrow(ax,[(.39,.86),(.39,.885),(.985,.885),(.985,.39),(.81,.39),(.81,.365)],color="blue")
    ax.text(.61,.862,"576 spatial tokens",fontsize=10,color=C["blue"])
    arrow(ax,[(.53,.27),(.59,.27),(.59,.295),(.775,.295)],color="teal")
    ax.text(.625,.312,"First 4 card tokens",fontsize=10,color=C["teal"])
    box(ax,.775,.075,.19,.09,"Mask + sample","P(slot) P(tile | slot)",color="teal",fs=11)
    arrow(ax,[(.87,.235),(.87,.165)],color="teal")
    ax.text(.60,.045,"Actor and Critic keep independent per-battle hidden / cell states.",fontsize=10,color=C["muted"])
    save(fig,"fig02_model_architecture")


def supervision():
    fig, ax = canvas(14, 9, "03  Auxiliary supervision and information boundary",
                     "Future targets supervise the network after collection; they are never policy observations")
    box(ax,.035,.70,.25,.12,"Observation at t","Public state + own hand",fs=12)
    box(ax,.365,.70,.25,.12,"Shared encoder + Actor LSTM","Actor latent u(t); entity tokens",fs=11)
    box(ax,.72,.70,.245,.12,"Masked policy action","Native execution -> state at t+1",color="teal",fs=11)
    arrow(ax,[(.285,.76),(.365,.76)])
    arrow(ax,[(.615,.76),(.72,.76)],color="teal")
    box(ax,.035,.45,.25,.12,"State-only heads","Elixir / hand / next-card belief",color="purple",fs=11)
    box(ax,.365,.45,.25,.12,"Action-conditioned heads","Tower / entities / damage / future",color="purple",fs=10)
    box(ax,.72,.45,.245,.12,"Executed action encoding","Wait + card one-hot + submitted x/y",color="orange",fs=10)
    arrow(ax,[(.42,.70),(.42,.63),(.16,.63),(.16,.57)],color="purple")
    arrow(ax,[(.49,.70),(.49,.57)],color="purple")
    arrow(ax,[(.845,.70),(.845,.57)],color="orange",dashed=True)
    arrow(ax,[(.72,.51),(.615,.51)],color="orange",dashed=True)
    box(ax,.035,.18,.25,.13,"Public teacher targets","Current reconstructed elixir / belief",color="orange",fs=10)
    box(ax,.365,.18,.25,.13,"Transition / future targets","ID-matched changes; t+4 proxy",color="orange",fs=11)
    box(ax,.72,.18,.245,.13,"Joint loss + backward","PPO + weighted auxiliary tasks",color="purple",fs=11)
    # Predictions and teacher targets independently enter the loss. Predictions
    # do not generate the labels; current labels derive from observation and
    # transition/future labels derive from the subsequently captured trajectory.
    arrow(ax,[(.035,.76),(.015,.76),(.015,.245),(.035,.245)],color="orange",dashed=True)
    arrow(ax,[(.965,.76),(.997,.76),(.997,.37),(.49,.37),(.49,.31)],color="orange",dashed=True)
    arrow(ax,[(.285,.51),(.335,.51),(.335,.34),(.68,.34),(.68,.28),(.72,.28)],color="purple")
    arrow(ax,[(.615,.51),(.65,.51),(.65,.30),(.72,.30)],color="purple")
    arrow(ax,[(.285,.245),(.32,.245),(.32,.12),(.66,.12),(.66,.245),(.72,.245)],color="orange",dashed=True)
    arrow(ax,[(.615,.245),(.72,.245)],color="orange",dashed=True)
    arrow(ax,[(.965,.245),(.985,.245),(.985,.85),(.49,.85),(.49,.82)],color="purple",dashed=True)
    ax.text(.035,.05,"Training-only channels: info -> rollout buffer -> padded sequence minibatch",fontsize=12,color=C["orange"])
    save(fig,"fig03_supervision_boundary")


def example():
    fig = plt.figure(figsize=(15,11),facecolor="white")
    fig.text(.04,.96,"04  Worked example: opponent play -> next decision -> native action",fontsize=18,weight="bold",color=C["ink"])
    fig.text(.04,.925,"Constructed state; assumed probabilities and placement. No trained-policy or native-battle result is shown.",fontsize=11,color=C["muted"])
    fig.text(.05,.86,"Previous boundary: tick 490\nOpponent Giant accepted: tick 491\nLearner observes the result: tick 500",fontsize=12,color=C["ink"],linespacing=1.6)
    a=fig.add_axes([.055,.19,.31,.59])
    a.set(xlim=(0,18),ylim=(0,32),xlabel="x (tiles)",ylabel="y (tiles)")
    a.set_aspect("equal")
    a.set_xticks(np.arange(0,19,3)); a.set_yticks(np.arange(0,33,4))
    a.set_xticks(np.arange(19),minor=True);a.set_yticks(np.arange(33),minor=True)
    a.grid(which="minor",color="#dde3eb",lw=.5)
    a.axhspan(15,17,color="#d8ebf8",alpha=.9)
    a.text(10,16,"River",va="center",color=C["blue"],fontsize=10)
    a.add_patch(Rectangle((3,12),1,1,fc="#bbddd5",ec=C["teal"],lw=2,zorder=3))
    a.add_patch(Circle((3.5,18.5),.47,color=C["red"],zorder=4))
    a.add_patch(Rectangle((2.9,5.9),1.2,1.2,fc=C["blue"],zorder=4))
    a.annotate("Enemy Giant\n(3.5, 18.5)",xy=(4,18.5),xytext=(7,21.5),fontsize=10,color=C["red"],arrowprops={"arrowstyle":"-","color":C["red"]})
    a.annotate("Own tower\n(3.5, 6.5)",xy=(4.1,6.5),xytext=(7,5.2),fontsize=10,color=C["blue"],arrowprops={"arrowstyle":"-","color":C["blue"]})
    a.annotate("Assumed placement\n(3.5, 12.5)",xy=(3.5,12.5),xytext=(7,9.2),fontsize=10,color=C["teal"],arrowprops={"arrowstyle":"-","color":C["teal"]})
    a.set_title("Egocentric 18 x 32 arena",fontsize=12,pad=14)
    fig.text(.05,.065,"Own hand: Mini P.E.K.K.A / Giant / Fireball / Archers\nOwn elixir: 4; next card: Knight\nGiant HP 3,000 / 3,000 and deployment 500 ms are illustrative.",fontsize=10.5,color=C["muted"],linespacing=1.6)
    ax=fig.add_axes([.415,.075,.56,.82]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    box(ax,.015,.84,.97,.12,"1  Confirm play -> update public ledger",
        "Giant: cost 5, event tick 491; capture tick 500 (20.50 s)\nOpponent elixir estimate: 10 - 5 + 0.45 / 2.8 = 5.160714",fs=12)
    box(ax,.015,.68,.97,.12,"2  Build complementary observations",
        "Spatial cell [18, 3] + entity + history + temporal changes\nHand [8, 6, 5, 2, 1]; Giant belief = 0; other hand beliefs = 4/7",fs=12)
    box(ax,.015,.52,.97,.12,"3  Encode context -> carry recurrent memory",
        "CNN: 576 x 64; entities -> history -> belief -> card queries\n384-d summaries -> 256-d fusion -> persistent Actor / Critic LSTMs",fs=12)
    box(ax,.015,.36,.97,.12,"4  Mask -> select slot -> conditional placement",
        "Assumed P(wait, Mini, Giant, Fireball, Archers) = (.18, .60, 0, .08, .14)\nFour 32 x 18 heatmaps; assume Mini's best legal tile is [12, 3]",color="teal",fs=12)
    box(ax,.015,.20,.97,.12,"5  Convert action -> execute -> next observation",
        "[1, 12, 3] -> Play(owner=0, slot=0, x=3500, y=12500)\nDeploy next tick; advance to tick 510; keep per-slot recurrent memory",color="teal",fs=12)
    box(ax,.015,.04,.97,.12,"6  Training only: supervise observed consequences",
        "Accepted card + submitted coordinates; ID-matched transition labels\nStore labels separately; combine PPO and auxiliary loss after rollout",color="purple",fs=12)
    for y in (.84,.68,.52,.36):arrow(ax,[(.5,y),(.5,y-.04)])
    arrow(ax,[(.5,.20),(.5,.16)],color="purple",dashed=True)
    save(fig,"fig04_worked_example")


def experiments():
    fig, ax=canvas(14,8,"05  Planned equal-interaction ablation protocol",
                    "Protocol only: the figure contains no measured rewards, win rates or experimental results")
    cols=[.365,.565,.765]
    for x,seed in zip(cols,(0,1,2)):
        ax.text(x+.075,.80,f"Training seed {seed}",ha="center",fontsize=13,weight="bold",color=C["ink"])
    variants=[("full (v6)","Action-conditioned auxiliary tasks","purple"),
              ("no_aux (v6)","Same parameters; auxiliary coefficient = 0","blue"),
              ("state_only (v5)","Auxiliary tasks without action input","teal"),
              ("v4","Spatial recurrent PPO; no auxiliary heads","orange")]
    for y,(name,detail,c) in zip((.62,.47,.32,.17),variants):
        box(ax,.035,y,.29,.105,name,detail,color=c,fs=11)
        for x in cols:box(ax,x,y,.15,.105,"819,200","learner decisions",color=c,fs=11)
    ax.text(.04,.085,"Development: 16 seeds x 2 roles x 2 opponents = 64 games / checkpoint",fontsize=12,color=C["muted"])
    ax.text(.04,.040,"Held-out test: 64 seeds x 2 roles x 2 opponents = 256 games / final checkpoint",fontsize=12,color=C["ink"])
    save(fig,"fig05_experiment_protocol")


if __name__ == "__main__":
    system(); architecture(); supervision(); example(); experiments()
    print(f"Wrote 5 SVG + 5 PNG figures to {OUT}")
