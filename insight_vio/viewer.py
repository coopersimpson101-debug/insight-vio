"""Minimal dependency-free 3D trajectory viewer.

* :func:`write_html` renders one or more trajectories into a self-contained HTML
  file (canvas + ~80 lines of JS: orbit with mouse drag, zoom with wheel).
* :func:`serve_live` runs a tiny HTTP server that listens for UDP pose packets
  and lets the same page poll ``/poses`` to draw the live head trajectory and the
  current headset frame.

No matplotlib / three.js needed, so it works on a headless Pi (open the page from
a laptop on the same network).
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>insight-vio viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafafa;--fg:#222;--grid:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--grid:#333}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
#info{position:absolute;left:12px;top:8px} canvas{display:block;width:100vw;height:100vh}
.k{display:inline-block;width:10px;height:10px;margin:0 4px 0 12px;border-radius:2px}
</style></head><body><div id="info">__TITLE__ <span id="legend"></span>
<div id="stat"></div></div><canvas id="c"></canvas><script>
const COLORS=["#1f77b4","#d62728","#2ca02c","#9467bd"];
let DATA=__DATA__; const LIVE=__LIVE__;
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let yaw=0.6,pitch=0.5,zoom=1,drag=null;
function resize(){cv.width=innerWidth*devicePixelRatio;cv.height=innerHeight*devicePixelRatio;draw();}
addEventListener('resize',resize);
cv.onmousedown=e=>drag=[e.clientX,e.clientY];onmouseup=()=>drag=null;
onmousemove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.01;pitch+=(e.clientY-drag[1])*0.01;
pitch=Math.max(-1.5,Math.min(1.5,pitch));drag=[e.clientX,e.clientY];draw();};
cv.onwheel=e=>{zoom*=Math.exp(-e.deltaY*0.001);draw();e.preventDefault();};
function bounds(){let lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];
for(const t of DATA.trajs)for(const p of t.p)for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}
if(lo[0]>hi[0])return [[0,0,0],1];const c=lo.map((l,i)=>(l+hi[i])/2);
return [c,Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],0.5)];}
function proj(p,c,s){const x=p[0]-c[0],y=p[1]-c[1],z=p[2]-c[2];
const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const x1=cy*x+sy*y,y1=-sy*x+cy*y; const y2=cp*y1-sp*z,z2=sp*y1+cp*z;
const f=Math.min(cv.width,cv.height)*0.8*zoom/s, d=3+y2/s;
return [cv.width/2+f*x1/d*2.5, cv.height/2-f*z2/d*2.5];}
function draw(){const [c,s]=bounds();ctx.fillStyle=getComputedStyle(document.body).backgroundColor;
ctx.fillRect(0,0,cv.width,cv.height);ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--grid');
ctx.lineWidth=1;const g=Math.pow(10,Math.floor(Math.log10(s)));
for(let i=-5;i<=5;i++){for(const [a,b] of [[[i*g,-5*g],[i*g,5*g]],[[-5*g,i*g],[5*g,i*g]]]){
const p0=proj([c[0]+a[0],c[1]+a[1],c[2]-s/2],c,s),p1=proj([c[0]+b[0],c[1]+b[1],c[2]-s/2],c,s);
ctx.beginPath();ctx.moveTo(...p0);ctx.lineTo(...p1);ctx.stroke();}}
let leg="";DATA.trajs.forEach((t,k)=>{ctx.strokeStyle=COLORS[k%4];ctx.lineWidth=2*devicePixelRatio;ctx.beginPath();
t.p.forEach((p,i)=>{const q=proj(p,c,s);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.stroke();
leg+=`<span class="k" style="background:${COLORS[k%4]}"></span>${t.name}`;
if(t.R&&t.p.length){const p=t.p[t.p.length-1],R=t.R;["#e33","#3a3","#36e"].forEach((col,a)=>{
const e=[p[0]+0.2*R[0][a],p[1]+0.2*R[1][a],p[2]+0.2*R[2][a]];ctx.strokeStyle=col;ctx.beginPath();
ctx.moveTo(...proj(p,c,s));ctx.lineTo(...proj(e,c,s));ctx.stroke();});}});
document.getElementById('legend').innerHTML=leg;document.getElementById('stat').textContent=DATA.stat||"";}
if(LIVE)setInterval(()=>fetch('/poses').then(r=>r.json()).then(d=>{DATA=d;draw();}).catch(()=>{}),100);
resize();
</script></body></html>"""


def _traj_json(trajs: dict, stat: str = "", R_last=None) -> dict:
    out = {"trajs": [], "stat": stat}
    for name, p in trajs.items():
        entry = {"name": name, "p": np.asarray(p, float).round(4).tolist()}
        if R_last is not None and name == next(iter(trajs)):
            entry["R"] = np.asarray(R_last).round(4).tolist()
        out["trajs"].append(entry)
    return out


def render_html(trajs: dict, title: str = "insight-vio", stat: str = "", live: bool = False) -> str:
    data = json.dumps(_traj_json(trajs, stat))
    return (_PAGE.replace("__TITLE__", title).replace("__DATA__", data)
            .replace("__LIVE__", "true" if live else "false"))


def write_html(path: str, trajs: dict, title: str = "insight-vio", stat: str = "") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(trajs, title, stat))


def serve_live(http_port: int = 8080, udp_port: int = 5555, max_points: int = 3000,
               host: str = "127.0.0.1"):  # pragma: no cover
    from .udp import packet_rotation, receive_forever
    pts, last = [], {"R": np.eye(3), "pkt": None}
    lock = threading.Lock()

    def on_packet(pkt):
        with lock:
            pts.append(pkt.position.copy())
            del pts[:-max_points]
            last["R"] = packet_rotation(pkt)
            last["pkt"] = pkt

    threading.Thread(target=receive_forever, args=(udp_port, host, on_packet),
                     daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/poses"):
                with lock:
                    pk = last["pkt"]
                    stat = "" if pk is None else (f"seq {pk.seq}  pos std {pk.pos_std:.3f} m  "
                                                  f"flags 0x{pk.flags:x}")
                    body = json.dumps(_traj_json({"live": list(pts)}, stat, last["R"])).encode()
                ctype = "application/json"
            else:
                body = render_html({"live": []}, "insight-vio live", live=True).encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"viewer on http://{host}:{http_port}  (listening for UDP poses on {host}:{udp_port})")
    ThreadingHTTPServer((host, http_port), H).serve_forever()
