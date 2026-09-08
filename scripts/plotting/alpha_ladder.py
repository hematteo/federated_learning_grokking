"""One setup's α ladder: the curves, the T_grok(α) relationship, and the fit.

An α ladder answers two questions that want different forms. "Does each cell
grok, and what does the trajectory look like?" is change-over-time, one small
multiple per rung. "How does T_grok depend on α?" is a relationship between two
scalars, one chart. This renders both against the same data so the second is
auditable from the first.

    python scripts/plotting/alpha_ladder.py --dataset s5 --model groknet \
        --groups central_anchor,d_alpha_fine --out ladder.html
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from fedgrok.analysis.survival import summarize_survival        # noqa: E402
from grok_curves import SETUP_NAMES, _thin, infer_setup, series_for   # noqa: E402


def fit_models(alphas, times):
    """Exponential and power-law fits, each with its R^2 on log10(T).

    The power law is the form setup A follows (T ~ (α-α_c)^-γ). α_c is scanned
    rather than assumed, and reported even when it runs to the search bound --
    hitting the bound is the diagnostic that the data does not constrain it.
    """
    a = np.asarray(alphas, float)
    y = np.log10(np.asarray(times, float))

    def r2(pred):
        return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    m, c = np.polyfit(a, y, 1)
    expo = {"kind": "exponential", "k": float(-m), "c": float(c),
            "r2": r2(m * a + c), "params": 2}

    best, lo, hi = None, 0.0, a.min() - 0.005
    for ac in np.arange(lo, hi, 0.0005):
        xs = np.log10(a - ac)
        g, cc = np.polyfit(xs, y, 1)
        rr = r2(g * xs + cc)
        if best is None or rr > best["r2"]:
            best = {"kind": "power", "alpha_c": float(ac), "gamma": float(-g),
                    "c": float(cc), "r2": rr, "params": 3,
                    "at_bound": bool(ac <= lo + 1e-9)}
    return expo, best


def build(csv_path, hist_root, dataset, model, groups, max_points):
    rows = [r for r in csv.DictReader(open(csv_path))
            if r["dataset"] == dataset and r["model"] == model
            and r["mode"] == "centralized"
            and (not groups or r.get("group") in groups)]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[float(r["alpha"])].append(r)

    rungs = []
    for alpha in sorted(cells, reverse=True):
        runs = cells[alpha]
        durations, events = [], []
        for r in runs:
            t = r["t_grok"]
            if t == "inf":
                durations.append(float(r["steps_run"] or 0)); events.append(0)
            else:
                durations.append(float(t)); events.append(1)
        s = summarize_survival(durations, events, n_boot=800)
        km = s["t_grok_km_median"]
        series = []
        for r in runs:
            hits = glob.glob(os.path.join(hist_root, r["id"], "history_*.json"))
            if not hits:
                continue
            with open(hits[0]) as fh:
                history = json.load(fh)
            xs, ys = series_for(history)
            if xs:
                series.append({k: [[int(x), round(v, 1)] for x, v in
                                   _thin(xs, vals, max_points)]
                               for k, vals in ys.items()})
        rungs.append({
            "alpha": alpha, "n": s["n_seeds"], "grokked": s["n_grokked"],
            "km": None if km == float("inf") else km,
            "ci": [s["t_grok_ci_low"], s["t_grok_ci_high"]] if s["n_grokked"] else None,
            "budget": max(durations) if durations else None,
            "threshold": float(runs[0]["grok_threshold"]),
            # Finite values only: alpha>=1.0 rows carry final_acc = nan (no
            # test set), and max() over a NaN-containing sequence returns a
            # position-dependent answer.
            "final_test": max((v for v in (float(r["final_acc"]) for r in runs)
                               if math.isfinite(v)), default=float("nan")),
            "series": series,
        })

    # alpha is the fraction of the grid used for TRAINING, so alpha=1.0 leaves no
    # test set at all, and compute_accuracy over zero samples returns NaN rather
    # than raising. Such a run does NOT look like a censored cell, which is what
    # this comment used to claim -- it looks like the opposite. NaN never
    # compared below the bar, so the old sustained-crossing scan reported the bar
    # as held from step 0 and banked `grokked=True, t_grok=0`. Five setup-D rows
    # are recorded that way and are still in the CSV. `split_indices` now rejects
    # the spec and `compute_t_grok` treats NaN as below the bar, so no new run
    # can join them; the flag keeps the banked five visibly marked.
    for r in rungs:
        r["no_test"] = r["alpha"] >= 1.0

    full = [r for r in rungs if r["grokked"] == r["n"] and r["km"]]
    fits = None
    if len(full) >= 3:
        expo, power = fit_models([r["alpha"] for r in full], [r["km"] for r in full])
        fits = {"exponential": expo, "power": power}
    return {
        "setup": infer_setup(rows[0]),
        "setup_name": SETUP_NAMES.get(infer_setup(rows[0]), f"{dataset}/{model}"),
        "budget": int(max(float(r["epochs"]) for r in rows)),
        "rungs": rungs, "fits": fits,
        "n_runs": len(rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/data/runs_v2.csv")
    ap.add_argument("--hist-root", default="results/runs")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--groups", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=90)
    ap.add_argument("--no-fit", action="store_true",
                    help="Curves only — drop the T_grok(α) chart and the fits.")
    args = ap.parse_args()

    data = build(args.csv, args.hist_root, args.dataset, args.model,
                 [g for g in args.groups.split(",") if g], args.max_points)
    if args.no_fit:
        data["fits"] = None
    html = _TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"  {len(data['rungs'])} rungs, {data['n_runs']} runs")
    if data["fits"]:
        e, p = data["fits"]["exponential"], data["fits"]["power"]
        print(f"  exponential k={e['k']:.3f} R2={e['r2']:.4f}")
        print(f"  power alpha_c={p['alpha_c']:.3f} gamma={p['gamma']:.2f} "
              f"R2={p['r2']:.4f}{'  (alpha_c AT SEARCH BOUND)' if p['at_bound'] else ''}")
    print(f"  -> {args.out} ({len(html)/1e6:.2f} MB)")


_TEMPLATE = r"""<title>α ladder — quadratic MLP on S₅</title>
<style>
  .viz-root{
    color-scheme:light;
    --surface-1:#fcfcfb; --surface-2:#f4f4f1; --border:#e3e2dd;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#82817b;
    --grid:#e8e7e2; --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
    --serif:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    font-family:var(--sans);background:var(--surface-1);color:var(--text-primary);
    padding:34px 22px 66px;max-width:1280px;margin:0 auto;line-height:1.5;
  }
  @media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19;--surface-2:#242423;--border:#3a3a38;
    --text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
    --grid:#2f2f2d;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
  }}
  :root[data-theme="dark"] .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19;--surface-2:#242423;--border:#3a3a38;
    --text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
    --grid:#2f2f2d;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;
  }
  .viz-root :focus-visible{outline:2px solid var(--series-1);outline-offset:3px;border-radius:4px}
  @media (prefers-reduced-motion:reduce){.viz-root *,.tipbox{transition:none!important}}
  h1{font-family:var(--serif);font-size:26px;font-weight:600;margin:0 0 8px;
     letter-spacing:-.012em;text-wrap:balance}
  .sub{font-size:14.5px;color:var(--text-secondary);margin:0 0 22px;max-width:72ch}
  .sub b{color:var(--text-primary);font-weight:600}
  .strip{display:flex;flex-wrap:wrap;border:1px solid var(--border);border-radius:10px;
    overflow:hidden;margin-bottom:26px;background:var(--surface-2)}
  .stat{flex:1 1 156px;padding:13px 16px;background:var(--surface-1);
    border-right:1px solid var(--border)}
  .stat:last-child{border-right:0}
  .stat .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
    color:var(--text-muted);font-weight:600;margin-bottom:3px}
  .stat .v{font-family:var(--serif);font-size:22px;font-weight:600;line-height:1.15}
  .stat .v small{font-size:12.5px;font-weight:400;color:var(--text-secondary);
    font-family:var(--sans)}
  h2{font-family:var(--serif);font-size:18px;font-weight:600;margin:34px 0 3px}
  .lead{font-size:13.5px;color:var(--text-secondary);margin:0 0 14px;max-width:72ch}
  .panel{border:1px solid var(--border);border-radius:10px;padding:16px 18px 10px;
    background:var(--surface-1)}
  .legend{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:6px}
  .lg{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--text-secondary)}
  .lg i{width:18px;height:2px;border-radius:1px;flex:none}
  .lg i.hollow{height:9px;width:9px;border-radius:99px;background:transparent;
    border:1.6px solid var(--text-muted)}
  svg{display:block;width:100%;height:auto;overflow:visible;touch-action:none}
  .grid{display:grid;gap:13px;grid-template-columns:repeat(auto-fill,minmax(216px,1fr));
    margin-top:14px}
  .card{border:1px solid var(--border);border-radius:9px;padding:10px 11px 7px}
  .card .t{font-family:var(--mono);font-size:11.5px;margin-bottom:1px}
  .card .s{font-size:10.5px;color:var(--text-muted);margin-bottom:3px;font-family:var(--mono)}
  .tick{font-size:10px;fill:var(--text-muted)}
  .axlab{font-size:11px;fill:var(--text-secondary)}
  .flab{font-size:11px;font-weight:600}
  .tablewrap{overflow-x:auto;margin-top:14px}
  table{border-collapse:collapse;width:100%;font-size:13px;min-width:660px;
    font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:7px 11px;border-bottom:1px solid var(--border);white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  th{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
    color:var(--text-muted);font-weight:600}
  tr.cens td{color:var(--text-muted)}
  .tipbox{position:fixed;pointer-events:none;z-index:50;background:var(--surface-1);
    border:1px solid var(--border);border-radius:8px;padding:9px 11px;font-size:12px;
    box-shadow:0 4px 18px rgba(0,0,0,.17);opacity:0;transition:opacity .1s;min-width:140px}
  .tipbox .x{color:var(--text-muted);font-size:11px;margin-bottom:5px;font-family:var(--mono)}
  .tipbox .r{display:flex;align-items:center;gap:8px;margin-top:3px}
  .tipbox .r i{width:13px;height:2px;border-radius:1px;flex:none}
  .tipbox .r b{margin-left:auto;font-family:var(--mono);font-weight:600}
  .note{font-size:12.5px;color:var(--text-muted);margin-top:26px;line-height:1.65;
    max-width:76ch;border-top:1px solid var(--border);padding-top:15px}
  .note b{color:var(--text-secondary)} .note p{margin:0 0 9px}
</style>

<div class="viz-root">
  <h1 id="h1"></h1>
  <p class="sub" id="sub"></p>
  <div class="strip" id="strip"></div>

  <div id="fitsec">
    <h2>How T<sub>grok</sub> depends on α</h2>
    <p class="lead" id="lead2"></p>
    <div class="panel">
      <div class="legend" id="leg2"></div>
      <svg id="fit" viewBox="0 0 720 330" role="img"
           aria-label="Grok time against training fraction, with fitted models."></svg>
    </div>
  </div>

  <h2>Every rung</h2>
  <p class="lead">Train (blue) and test (orange) accuracy against epochs, all five
  seeds drawn. The faint horizontal rule is S₅'s 85% grok threshold.</p>
  <div class="grid" id="cards"></div>

  <h2>Table</h2>
  <div class="tablewrap"><table id="tab"></table></div>
  <div class="note" id="note"></div>
</div>
<div class="tipbox" id="tip" role="status" aria-live="polite"></div>

<script>
const D = __DATA__;
const tip=document.getElementById("tip");
const NS="http://www.w3.org/2000/svg";
const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
const fmt=v=>v>=1000?(v/1000).toFixed(v>=10000?0:1)+"k":String(Math.round(v));

document.getElementById("h1").textContent =
  `α ladder — ${D.setup} · ${D.setup_name}`;
// Built as nodes rather than markup strings: keeps the no-innerHTML rule
// absolute, so nothing that later flows in from a CSV header or a tool can
// become markup by accident.
function rich(el, parts){
  el.replaceChildren();
  parts.forEach(p=>{
    if(typeof p==="string"){el.appendChild(document.createTextNode(p)); return;}
    const t=document.createElement(p.tag||"b"); t.textContent=p.text; el.appendChild(t);
  });
}
rich(document.getElementById("sub"), [
  `Centralized, ${D.n_runs} runs over ${D.rungs.length} values of α, five seeds each, `,
  `${D.budget.toLocaleString()}-epoch budget. α is the fraction of the grid used for `,
  `training. `, {text:"Grokking is the gap"},
  ` between train (blue) saturating and test (orange) following.`]);

// ── verdict strip
const full=D.rungs.filter(r=>r.grokked===r.n&&r.km);
const cens=D.rungs.filter(r=>r.grokked===0&&!r.no_test);
const e=D.fits&&D.fits.exponential, p=D.fits&&D.fits.power;
[["Rungs", D.rungs.length, "values of α"],
 ["All seeds grokked", `${full.length}/${D.rungs.length}`, "cells at 5/5"],
 ["Cliff bracketed", cens.length?`${Math.max(...cens.map(r=>r.alpha))}–${Math.min(...full.map(r=>r.alpha))}`:"—", "between these α"],
 ["Fastest", `${Math.round(Math.min(...full.map(r=>r.km))).toLocaleString()}`, `epochs, at α=${full.reduce((a,b)=>a.km<b.km?a:b).alpha}`]
].forEach(([k,v,n])=>{
  const d=document.createElement("div"); d.className="stat";
  const kk=document.createElement("div"); kk.className="k"; kk.textContent=k;
  const vv=document.createElement("div"); vv.className="v";
  vv.append(document.createTextNode(String(v)));
  const s=document.createElement("small"); s.textContent=" "+n; vv.appendChild(s);
  d.append(kk,vv); document.getElementById("strip").appendChild(d);
});

if(!D.fits){ document.getElementById("fitsec").remove(); }
if(D.fits) rich(document.getElementById("lead2"), [
  "Points are Kaplan–Meier medians with bootstrap 95% intervals; y is log-scaled, so an ",
  "exponential in α is a straight line. The power law is the form the anchor setup follows — ",
  "here its α", {tag:"sub", text:"c"}, " fits to ", {text:p.alpha_c.toFixed(3)},
  p.at_bound
    ? ", the bottom of the search range: the data does not constrain it, and it contradicts the measured cliff. "
    : ". ",
  "Hollow markers are cells where no seed grokked; they sit at the budget, which is a lower bound, not a measurement."]);

// ── fit chart (omitted when fits are off)
if(D.fits) (function(){
  const svg=document.getElementById("fit");
  const W=720,H=330,ML=62,MR=104,MT=14,MB=42,PW=W-ML-MR,PH=H-MT-MB;
  const as=D.rungs.map(r=>r.alpha);
  const aMin=Math.min(...as)-0.02, aMax=Math.max(...as)+0.02;
  const tv=D.rungs.flatMap(r=>[r.km,r.budget,...(r.ci&&isFinite(r.ci[1])?[r.ci[1]]:[])]).filter(Boolean);
  const yMin=Math.min(...tv)*0.72, yMax=Math.max(...tv)*1.5;
  const X=a=>ML+PW*(a-aMin)/(aMax-aMin);
  const Y=t=>MT+PH*(1-(Math.log10(t)-Math.log10(yMin))/(Math.log10(yMax)-Math.log10(yMin)));

  for(let d=Math.ceil(Math.log10(yMin)); d<=Math.floor(Math.log10(yMax)); d++)
    for(const mlt of [1,2,5]){
      const v=mlt*10**d; if(v<yMin||v>yMax) continue;
      svg.appendChild(mk("line",{x1:ML,x2:ML+PW,y1:Y(v),y2:Y(v),stroke:"var(--grid)","stroke-width":1}));
      const t=mk("text",{x:ML-8,y:Y(v)+3.5,"text-anchor":"end"});
      t.setAttribute("class","tick"); t.textContent=fmt(v); svg.appendChild(t);
    }
  [0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55].forEach(a=>{
    if(a<aMin||a>aMax) return;
    const t=mk("text",{x:X(a),y:H-MB+18,"text-anchor":"middle"});
    t.setAttribute("class","tick"); t.textContent=a; svg.appendChild(t);
  });
  let l=mk("text",{x:ML+PW/2,y:H-6,"text-anchor":"middle"}); l.setAttribute("class","axlab");
  l.textContent="α  (training fraction)"; svg.appendChild(l);
  l=mk("text",{x:14,y:MT+PH/2,"text-anchor":"middle",transform:`rotate(-90 14 ${MT+PH/2})`});
  l.setAttribute("class","axlab"); l.textContent="T_grok  (epochs, log)"; svg.appendChild(l);

  // fitted models, drawn under the data
  const line=(fn,colour,label,dy)=>{
    let d="";
    for(let i=0;i<=120;i++){
      const a=aMin+(aMax-aMin)*i/120, v=fn(a);
      if(!isFinite(v)||v<=0) continue;
      const y=Y(Math.max(Math.min(v,yMax),yMin));
      d+=(d?"L":"M")+X(a).toFixed(1)+" "+y.toFixed(1);
    }
    svg.appendChild(mk("path",{d,fill:"none",stroke:colour,"stroke-width":2,
      "stroke-linecap":"round","stroke-opacity":.9}));
    const t=mk("text",{x:ML+PW+8,y:Y(fn(aMax-0.02))+dy});
    t.setAttribute("class","flab"); t.setAttribute("fill",colour);
    t.textContent=label; svg.appendChild(t);
  };
  line(a=>10**(-e.k*a+e.c), "var(--series-2)", `exponential  R²=${e.r2.toFixed(3)}`, -6);
  line(a=>a>p.alpha_c?10**p.c*Math.pow(a-p.alpha_c,-p.gamma):NaN,
       "var(--series-3)", `power law  R²=${p.r2.toFixed(3)}`, 12);

  D.rungs.forEach(r=>{
    const grok=r.grokked===r.n&&r.km;
    if(grok&&r.ci&&isFinite(r.ci[1])){
      svg.appendChild(mk("line",{x1:X(r.alpha),x2:X(r.alpha),y1:Y(r.ci[0]),y2:Y(r.ci[1]),
        stroke:"var(--series-1)","stroke-width":1.4,"stroke-opacity":.55}));
    }
    const cx=X(r.alpha), cy=Y(grok?r.km:r.budget);
    svg.appendChild(mk("circle",{cx,cy,r:grok?5:4.5,
      fill:grok?"var(--series-1)":"transparent",
      stroke:grok?"var(--surface-1)":"var(--text-muted)","stroke-width":grok?2:1.6}));
    if(!grok){  // censored: the true value is somewhere below
      svg.appendChild(mk("path",{d:`M${cx} ${cy+7}L${cx} ${cy+16}M${cx-3.5} ${cy+12}L${cx} ${cy+16}L${cx+3.5} ${cy+12}`,
        stroke:"var(--text-muted)","stroke-width":1.4,fill:"none"}));
    }
  });
  const hit=mk("rect",{x:ML,y:MT,width:PW,height:PH,fill:"transparent"}); svg.appendChild(hit);
  svg.addEventListener("pointermove",ev=>{
    const bb=svg.getBoundingClientRect();
    const a=aMin+(aMax-aMin)*(((ev.clientX-bb.left)/bb.width*W)-ML)/PW;
    let best=D.rungs[0];
    D.rungs.forEach(r=>{if(Math.abs(r.alpha-a)<Math.abs(best.alpha-a))best=r;});
    tip.replaceChildren();
    const x=document.createElement("div"); x.className="x";
    x.textContent=`α = ${best.alpha}`; tip.appendChild(x);
    const rows=[["grokked",`${best.grokked}/${best.n}`],
      ["T_grok", best.km?Math.round(best.km).toLocaleString():"censored"],
      ["peak test",best.final_test.toFixed(1)+"%"]];
    rows.forEach(([k,v])=>{const r=document.createElement("div"); r.className="r";
      const s=document.createElement("span"); s.textContent=k;
      const b=document.createElement("b"); b.textContent=v;
      r.append(s,b); tip.appendChild(r);});
    tip.style.opacity=1;
    tip.style.left=Math.min(ev.clientX+14,innerWidth-165)+"px";
    tip.style.top=Math.max(8,ev.clientY-56)+"px";
  });
  svg.addEventListener("pointerleave",()=>tip.style.opacity=0);
  [["var(--series-1)","Measured T_grok (KM median, 95% CI)"],
   ["var(--series-2)","Exponential fit"],["var(--series-3)","Power-law fit"]]
   .forEach(([c,t])=>{const s=document.createElement("span"); s.className="lg";
     const i=document.createElement("i"); i.style.background=c;
     s.append(i,document.createTextNode(t)); document.getElementById("leg2").appendChild(s);});
  const s=document.createElement("span"); s.className="lg";
  const i=document.createElement("i"); i.className="hollow";
  s.append(i,document.createTextNode("No seed grokked (plotted at budget)"));
  document.getElementById("leg2").appendChild(s);
})();

// ── per-rung curves
const W2=232,H2=126,ML2=24,MR2=7,MT2=7,MB2=17,PW2=W2-ML2-MR2,PH2=H2-MT2-MB2;
D.rungs.forEach(r=>{
  const el=document.createElement("div"); el.className="card";
  const t=document.createElement("div"); t.className="t"; t.textContent=`α = ${r.alpha}`;
  const s=document.createElement("div"); s.className="s";
  s.textContent = r.no_test
    ? `${r.grokked}/${r.n}  ·  no test set`
    : `${r.grokked}/${r.n}` + (r.km?`  ·  T=${Math.round(r.km).toLocaleString()}`:"  ·  censored");
  el.append(t,s);
  let lo=Infinity,hi=0;
  r.series.forEach(se=>["train_acc","test_acc"].forEach(k=>(se[k]||[]).forEach(([x])=>{
    if(x<lo)lo=x; if(x>hi)hi=x;})));
  lo=Math.max(lo,1);
  const L0=Math.log10(lo),L1=Math.log10(Math.max(hi,lo*10));
  const X=v=>ML2+PW2*(Math.log10(Math.max(v,lo))-L0)/((L1-L0)||1);
  const Y=v=>MT2+PH2*(1-Math.max(0,Math.min(100,v))/100);
  const svg=mk("svg",{viewBox:`0 0 ${W2} ${H2}`,role:"img",tabindex:"0"});
  svg.setAttribute("aria-label",`alpha ${r.alpha}, ${r.grokked} of ${r.n} seeds grokked.`);
  [0,50,100].forEach(v=>{
    svg.appendChild(mk("line",{x1:ML2,x2:W2-MR2,y1:Y(v),y2:Y(v),stroke:"var(--grid)","stroke-width":1}));
    const q=mk("text",{x:ML2-4,y:Y(v)+3,"text-anchor":"end"});
    q.setAttribute("class","tick"); q.textContent=v; svg.appendChild(q);});
  svg.appendChild(mk("line",{x1:ML2,x2:W2-MR2,y1:Y(r.threshold),y2:Y(r.threshold),
    stroke:"var(--text-muted)","stroke-width":1,"stroke-opacity":.45}));
  for(let d=Math.ceil(L0);d<=Math.floor(L1);d++){
    const q=mk("text",{x:X(10**d),y:H2-4,"text-anchor":"middle"});
    q.setAttribute("class","tick");
    q.textContent=(10**d>=1000)?(10**d/1000)+"k":10**d; svg.appendChild(q);}
  r.series.forEach(se=>[["train_acc","--series-1"],["test_acc","--series-2"]].forEach(([k,c])=>{
    const pts=se[k]; if(!pts) return;
    const d=pts.map(([x,y],i)=>(i?"L":"M")+X(x).toFixed(1)+" "+Y(y).toFixed(1)).join(" ");
    svg.appendChild(mk("path",{d,fill:"none",stroke:`var(${c})`,"stroke-width":1.4,
      "stroke-opacity":.82,"stroke-linejoin":"round","stroke-linecap":"round"}));}));
  el.appendChild(svg); document.getElementById("cards").appendChild(el);
});

// ── table (the relief the contrast WARN requires, and the tooltip's twin)
const tb=document.getElementById("tab");
const hr=document.createElement("tr");
(D.fits?["α","seeds grokked","T_grok (KM)","95% CI","peak test","exponential fit","power fit"]
       :["α","seeds grokked","T_grok (KM)","95% CI","peak test","note"])
  .forEach(h=>{const th=document.createElement("th"); th.textContent=h; hr.appendChild(th);});
tb.appendChild(document.createElement("thead")).appendChild(hr);
const tbody=document.createElement("tbody");
D.rungs.forEach(r=>{
  const tr=document.createElement("tr");
  if(!r.km) tr.className="cens";
  const base=[r.alpha, `${r.grokked}/${r.n}`,
   r.no_test?"n/a":(r.km?Math.round(r.km).toLocaleString():"censored"),
   r.ci&&isFinite(r.ci[1])?`${Math.round(r.ci[0]).toLocaleString()} – ${Math.round(r.ci[1]).toLocaleString()}`:"—",
   r.no_test?"n/a":r.final_test.toFixed(1)+"%"];
  let extra;
  if(D.fits){
    const ef=10**(-e.k*r.alpha+e.c);
    const pf=r.alpha>p.alpha_c?10**p.c*Math.pow(r.alpha-p.alpha_c,-p.gamma):NaN;
    extra=[Math.round(ef).toLocaleString(), isFinite(pf)?Math.round(pf).toLocaleString():"—"];
  } else {
    extra=[r.no_test?"no test set — α=1.00 trains on the whole grid":""];
  }
  base.concat(extra).forEach(v=>{const td=document.createElement("td"); td.textContent=v; tr.appendChild(td);});
  tbody.appendChild(tr);
});
tb.appendChild(tbody);

[["Censored rungs are lower bounds","A cell where no seed grokked is plotted at its budget with a downward arrow. The true T_grok is somewhere below that point or does not exist; it is not a measurement and is excluded from both fits."],
 ["The fits are on the fully-grokked rungs only","Both models are least squares on log₁₀ T_grok. The exponential has two free parameters, the power law three (α_c, γ, scale) — so the exponential winning on R² is not a complexity artefact."],
 ["Seeds are drawn, not averaged","Each rung panel shows five blue and five orange lines. Tight bundles mean the grok time is reproducible at that α; fanning means it is seed-dependent."]
].forEach(([h,t])=>{const q=document.createElement("p");
  const b=document.createElement("b"); b.textContent=h+". ";
  q.append(b,document.createTextNode(t)); document.getElementById("note").appendChild(q);});
</script>
"""


if __name__ == "__main__":
    main()
