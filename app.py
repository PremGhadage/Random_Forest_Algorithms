import os
import pickle
import warnings

import numpy as np
from flask import Flask, Response, jsonify, request

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "decision_1.pkl"), "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)
CLASSES = [str(c) for c in model.classes_]

# Categorical columns: number of encoded values (0..n-1).
CATS = {"company_name": 20, "industry": 7, "country": 6, "company_size": 4,
        "month": 12, "reason_for_layoffs": 5, "hiring_trend": 4, "top_hiring_role": 8}
# Optional real labels, in the SAME ORDER as your LabelEncoder classes_.
# Example: NAMES["industry"] = ["Banking", "Retail", ...]
NAMES = {
    "month": ["April", "August", "December", "February", "January", "July",
              "June", "March", "May", "November", "October", "September"],
}
LAB = {"company_name": "Company", "industry": "Industry", "country": "Country",
       "company_size": "Company size", "month": "Month", "year": "Year",
       "reason_for_layoffs": "Reason for layoffs", "hiring_trend": "Hiring trend",
       "top_hiring_role": "Top hiring role"}
YEARS = [2024, 2025]
# key: (label, min, max, step, typical value, unit)
NUM = {
    "layoffs_count": ("Layoffs (people)", 0, 15000, 50, 2400, ""),
    "layoff_percentage": ("Layoff percentage", 0, 30, 0.1, 9, "%"),
    "ai_automation_impact": ("AI automation impact", 0, 17, 0.1, 6, "%"),
    "ai_replacement_risk": ("AI replacement risk (1-10)", 1, 10, 0.1, 6, ""),
    "open_roles": ("Open roles", 0, 10000, 50, 2300, ""),
    "remote_jobs_percentage": ("Remote jobs", 10, 90, 0.5, 50, "%"),
    "stock_growth_percent": ("Stock growth", -45, 90, 0.5, 22, "%"),
    "revenue_growth_percent": ("Revenue growth", -25, 60, 0.5, 14, "%"),
    "salary_budget_change": ("Salary budget change", -20, 35, 0.5, 6.5, "%"),
    "ai_adoption_level": ("AI adoption level (1-10)", 1, 10, 0.1, 5.8, ""),
    "employee_sentiment": ("Employee sentiment (1-10)", 3, 10, 0.1, 6.9, ""),
    "job_security_score": ("Job security score (1-10)", 2.5, 9.5, 0.1, 6.3, ""),
}
GROUPS = [
    ("Company", ["company_name", "industry", "country", "company_size", "month", "year"]),
    ("Layoffs and AI", ["layoffs_count", "layoff_percentage", "reason_for_layoffs",
                        "ai_automation_impact", "ai_replacement_risk"]),
    ("Hiring", ["open_roles", "hiring_trend", "remote_jobs_percentage", "top_hiring_role"]),
    ("Finance", ["stock_growth_percent", "revenue_growth_percent", "salary_budget_change"]),
    ("Workforce", ["ai_adoption_level", "employee_sentiment", "job_security_score"]),
]
MEANING = {
    "Bull Market": "strong growth, with healthy hiring, rising revenue and confident employees.",
    "Stable": "steady conditions, with no strong boom or downturn signals.",
    "Recession": "contraction, with heavy layoffs, weak hiring and falling growth.",
}

app = Flask(__name__)


def options(k):
    if k == "year":
        return [{"v": y, "t": str(y)} for y in YEARS]
    names = NAMES.get(k) or [f"{LAB[k]} #{i}" for i in range(CATS[k])]
    return [{"v": i, "t": t} for i, t in enumerate(names)]


def field(k):
    if k in NUM:
        l, lo, hi, st, df, u = NUM[k]
        return dict(key=k, type="num", label=l, min=lo, max=hi, step=st, **{"def": df}, unit=u)
    return dict(key=k, type="sel", label=LAB[k], options=options(k))


def read_input(data):
    row = []
    for k in FEATURES:
        try:
            v = float(data.get(k))
        except (TypeError, ValueError):
            raise ValueError(f"Enter a number for {NUM[k][0] if k in NUM else LAB.get(k, k)}.")
        if k in NUM:
            if not NUM[k][1] <= v <= NUM[k][2]:
                raise ValueError(f"{NUM[k][0]} must be between {NUM[k][1]} and {NUM[k][2]}.")
        elif k == "year":
            if v not in YEARS:
                raise ValueError("Choose a valid year.")
        elif not (v == int(v) and 0 <= v < CATS[k]):
            raise ValueError(f"Choose a valid option for {LAB[k]}.")
        row.append(v)
    return row


def drivers(row, top, base_p):
    """Compare each numeric input with a typical value to see what pushes the result."""
    idx = [i for i, k in enumerate(FEATURES) if k in NUM]
    M = np.tile(row, (len(idx), 1))
    for r, i in enumerate(idx):
        M[r, i] = NUM[FEATURES[i]][4]
    typ = model.predict_proba(M)[:, top]
    out = []
    for r, i in enumerate(idx):
        d = base_p - typ[r]
        if abs(d) >= 0.02:
            k = FEATURES[i]
            out.append((abs(d), f"{NUM[k][0]} of {row[i]:g}{NUM[k][5]} "
                        f"{'raises' if d > 0 else 'lowers'} the chance of {CLASSES[top]} "
                        f"by about {abs(d) * 100:.0f} points compared with a typical value ({NUM[k][4]:g}{NUM[k][5]})."))
    return [t for _, t in sorted(out, reverse=True)[:4]]


@app.get("/api/schema")
def schema():
    return jsonify(groups=[{"title": t, "fields": [field(k) for k in ks]} for t, ks in GROUPS])


@app.post("/api/predict")
def predict():
    try:
        row = read_input(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    p = model.predict_proba(np.array([row]))[0]
    top = int(p.argmax())
    conf = p[top] * 100
    level = "high" if conf >= 70 else "moderate" if conf >= 50 else "low"
    second = sorted(range(len(p)), key=lambda i: -p[i])[1]
    summary = (f"The model reads these inputs as a {CLASSES[top]} outlook, meaning "
               f"{MEANING.get(CLASSES[top], 'the pattern it learned for this class.')} "
               f"Confidence is {level} ({conf:.0f}%).")
    if level != "high":
        summary += f" The signals are mixed; {CLASSES[second]} is the next most likely ({p[second] * 100:.0f}%)."
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:6]
    return jsonify(
        label=CLASSES[top], confidence=round(conf, 1), summary=summary,
        probabilities=[{"name": c, "value": round(float(v) * 100, 1)} for c, v in zip(CLASSES, p)],
        drivers=drivers(row, top, p[top]),
        importance=[{"name": NUM[k][0] if k in NUM else LAB.get(k, k), "value": round(float(v) * 100, 1)}
                    for k, v in imp])


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Outlook Predictor</title>
<script>try{var t=localStorage.getItem("theme")||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");document.documentElement.dataset.theme=t}catch(e){document.documentElement.dataset.theme="light"}</script>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#f3f6f8;--card:#fff;--ink:#14202b;--mute:#5b6a77;--line:#dde4ea;--acc:#0b8a7a;--info:#2f6fd0;--bad:#d1434e;--track:#e5ebf0;color-scheme:light}
[data-theme=dark]{--bg:#0d1218;--card:#151c24;--ink:#e8eef3;--mute:#92a1ad;--line:#25303b;--acc:#2fc9b4;--info:#6ea8fe;--bad:#ff6b75;--track:#25303b;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:500 15px/1.5 Manrope,system-ui,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}
h1{font-size:1.5rem;margin:0;font-weight:800;letter-spacing:-.02em}header p{margin:2px 0 0;color:var(--mute)}
h2{font-size:1rem;margin:0 0 10px;font-weight:700}
button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 14px;font-weight:700}
button:hover{border-color:var(--acc)}button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
[data-theme=dark] button.primary{color:#06201b}
:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.grid{display:grid;grid-template-columns:1fr 390px;gap:16px;align-items:start}
@media(max-width:920px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin-bottom:14px}
summary{cursor:pointer;font-weight:800}
.opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin-top:14px}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px}
label{font-weight:700;font-size:.9rem}
input[type=range]{width:100%;accent-color:var(--acc);margin:6px 0 0}
input[type=number],select{font:inherit;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 8px}
input[type=number]{width:92px;text-align:right}select{width:100%;margin-top:4px}
.num{display:flex;align-items:center;gap:4px}
.chips{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px}.sp{flex:1}
.side{position:sticky;top:12px}@media(max-width:920px){.side{position:static}}
.ringbox{display:flex;align-items:center;gap:16px}
.ring{position:relative;width:120px;height:120px;flex:none}
.ring svg{transform:rotate(-90deg)}.ring circle{fill:none;stroke-width:10;stroke-linecap:round}
#arc{stroke:var(--acc);stroke-dasharray:326.7;stroke-dashoffset:326.7;transition:stroke-dashoffset .6s ease,stroke .3s}
#pct{position:absolute;inset:0;display:grid;place-items:center;font-size:1.6rem;font-weight:800}
#verdict{font-size:1.3rem;font-weight:800;margin:0}
ul{padding-left:18px;margin:6px 0 0}li{margin-bottom:6px}
.bar{display:grid;grid-template-columns:130px 1fr 46px;gap:8px;align-items:center;font-size:.85rem;margin-bottom:6px}
.bar div{background:var(--track);border-radius:8px}.bar i{height:8px;border-radius:8px;background:var(--c,var(--acc));width:var(--w);display:block;transition:width .5s}
.bar b{text-align:right}.err{color:var(--bad);font-weight:700}small{color:var(--mute)}
.hist div{display:flex;justify-content:space-between;gap:8px;padding:6px 0;border-bottom:1px solid var(--line);font-size:.88rem}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><div class="wrap">
<header><div><h1>Market Outlook Predictor</h1><p>Describe a company's layoffs, hiring and finances to see which outlook the model expects.</p></div>
<button id="theme" aria-label="Switch light or dark theme">Theme</button></header>
<div class="grid"><main>
<div class="chips"><span>Try a scenario:</span><button data-p="boom">Boom</button><button data-p="steady">Steady</button><button data-p="down">Downturn</button></div>
<div id="form"></div>
<div class="chips"><button class="primary" id="go">Predict</button><button id="reset">Reset</button><span class="sp"></span>
<label><input type="checkbox" id="live" checked> Update live as I adjust</label></div>
</main>
<aside class="side"><section class="card" aria-live="polite"><h2>Prediction</h2>
<div class="ringbox"><div class="ring"><svg width="120" height="120" viewBox="0 0 120 120"><circle cx="60" cy="60" r="52" stroke="var(--track)"/><circle id="arc" cx="60" cy="60" r="52"/></svg><div id="pct">--</div></div>
<div><p id="verdict">Loading...</p><small>Model confidence</small></div></div>
<p id="summary" style="margin-top:14px"></p><div id="body"></div></section>
<section class="card hist"><div class="row"><h2>Recent predictions</h2><button id="clr">Clear</button></div><div id="hist"><small>Nothing yet. Press Predict to save a result.</small></div></section>
</aside></div>
<p><small>This is a statistical estimate learned from past data, not financial advice.</small></p></div>
<script>
const $=s=>document.querySelector(s),COL={"Bull Market":"var(--acc)","Stable":"var(--info)","Recession":"var(--bad)"};
const PRE={boom:{layoffs_count:400,layoff_percentage:2,open_roles:8000,stock_growth_percent:60,revenue_growth_percent:40,salary_budget_change:20,job_security_score:8.5,employee_sentiment:8.5,ai_automation_impact:3,ai_replacement_risk:3},steady:{},down:{layoffs_count:9000,layoff_percentage:22,open_roles:500,stock_growth_percent:-30,revenue_growth_percent:-15,salary_budget_change:-10,job_security_score:3.5,employee_sentiment:4.5,ai_automation_impact:14,ai_replacement_risk:8}};
let S,timer,seq=0;
fetch("/api/schema").then(r=>r.json()).then(s=>{S=s;build();run(false)});
function fh(f){const c=f.type==="sel"?`<div><label for="f_${f.key}">${f.label}</label><select id="f_${f.key}" data-k="${f.key}">${f.options.map(o=>`<option value="${o.v}">${o.t}</option>`).join("")}</select></div>`
:`<div><div class="row"><label for="f_${f.key}">${f.label}</label><span class="num"><input type="number" id="f_${f.key}" data-k="${f.key}" data-n="1" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.def}">${f.unit}</span></div><input type="range" data-r="${f.key}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.def}" aria-label="${f.label}"></div>`;return c}
function build(){$("#form").innerHTML=S.groups.map((g,i)=>`<details class="card" ${i<3?"open":""}><summary>${g.title}</summary><div class="opts">${g.fields.map(fh).join("")}</div></details>`).join("");
document.querySelectorAll("[data-r]").forEach(r=>r.oninput=()=>{$("#f_"+r.dataset.r).value=r.value;sched()});
document.querySelectorAll("[data-n]").forEach(n=>n.oninput=()=>{const r=document.querySelector(`[data-r=${n.dataset.k}]`);if(r)r.value=n.value;sched()});
document.querySelectorAll("select").forEach(s=>s.onchange=sched)}
function setNum(k,v){$("#f_"+k).value=v;const r=document.querySelector(`[data-r=${k}]`);if(r)r.value=v}
function defaults(){S.groups.forEach(g=>g.fields.forEach(f=>{if(f.type==="num")setNum(f.key,f.def)}))}
function collect(){const d={};document.querySelectorAll("[data-k]").forEach(e=>d[e.dataset.k]=e.value);return d}
function sched(){if($("#live").checked){clearTimeout(timer);timer=setTimeout(()=>run(false),300)}}
async function run(save){
  const d=collect(),my=++seq;
  try{const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
    const r=await res.json();if(my!==seq)return;
    if(!res.ok){$("#summary").innerHTML=`<span class="err">${r.error||"Something went wrong."}</span>`;return}
    show(r);if(save)addH(r,d)}
  catch(e){$("#summary").innerHTML='<span class="err">Could not reach the server. Check your connection and try again.</span>'}}
function show(r){
  const c=COL[r.label]||"var(--info)";$("#arc").style.stroke=c;$("#arc").style.strokeDashoffset=326.7*(1-r.confidence/100);
  $("#pct").textContent=Math.round(r.confidence)+"%";$("#verdict").textContent=r.label;$("#verdict").style.color=c;$("#summary").textContent=r.summary;
  const mx=Math.max(...r.importance.map(i=>i.value),1);
  $("#body").innerHTML=`<h2 style="margin-top:14px">Chance of each outlook</h2>${r.probabilities.map(p=>`<div class="bar" style="--c:${COL[p.name]||"var(--info)"}"><span>${p.name}</span><div><i style="--w:${p.value}%"></i></div><b>${p.value}%</b></div>`).join("")}
  <h2 style="margin-top:16px">What is driving this</h2><ul>${(r.drivers.length?r.drivers:["No single input stands out; the result comes from many small signals together."]).map(x=>`<li>${x}</li>`).join("")}</ul>
  <h2 style="margin-top:16px">What the model weighs most overall</h2>${r.importance.map(i=>`<div class="bar"><span>${i.name}</span><div><i style="--w:${i.value/mx*100}%"></i></div><b>${i.value}%</b></div>`).join("")}`}
function getH(){try{return JSON.parse(localStorage.getItem("hist")||"[]")}catch(e){return[]}}
function drawH(h){$("#hist").innerHTML=h.length?h.map(x=>`<div><span>${x.l} (${x.p}%)</span><small>${x.s}</small></div>`).join(""):"<small>Nothing yet. Press Predict to save a result.</small>"}
function addH(r,d){const h=getH();h.unshift({p:Math.round(r.confidence),l:r.label,s:`${d.layoffs_count} layoffs, ${d.open_roles} open roles`});h.length=Math.min(h.length,5);try{localStorage.setItem("hist",JSON.stringify(h))}catch(e){}drawH(h)}
$("#clr").onclick=()=>{try{localStorage.removeItem("hist")}catch(e){}drawH([])};
$("#go").onclick=()=>run(true);
$("#reset").onclick=()=>{defaults();document.querySelectorAll("select").forEach(s=>s.selectedIndex=0);run(false)};
document.querySelectorAll("[data-p]").forEach(b=>b.onclick=()=>{defaults();Object.entries(PRE[b.dataset.p]).forEach(([k,v])=>setNum(k,v));run(false)});
$("#theme").onclick=()=>{const t=document.documentElement.dataset.theme==="dark"?"light":"dark";document.documentElement.dataset.theme=t;try{localStorage.setItem("theme",t)}catch(e){}};
drawH(getH());
</script></body></html>"""

if __name__ == "__main__":
    app.run(debug=True)