"""Animated 3D viewer of a crowded Martini cell-chunk trajectory.

Water is ~97% of the beads, so it is hidden by default (an optional faint
subsample can be shown). Renders the SOLUTE -- proteins (coloured by species)
and the chromosome (dna_segment) -- animated over the trajectory frames, with
play/pause + a frame slider. Self-contained HTML (three.js, frames embedded).

The build writes solute beads first in the .gro, so we take the first N_solute
atoms (derived from the .top [molecules] minus W/NA/CL).

Usage::

    python scripts/make_crowded_viewer.py --dcd traj.dcd --gro system.gro \
        --top system.top --itp-dir <chunk_dir> --out crowded.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

from viva_openmm.topology import _moleculetype_name, _moleculetype_natoms

_SOLVENT = {"W", "NA", "CL"}
_PALETTE = [
    [0.45, 0.80, 0.50], [0.35, 0.60, 0.95], [0.90, 0.40, 0.55],
    [0.95, 0.85, 0.30], [0.70, 0.55, 0.90], [0.40, 0.85, 0.85],
    [0.95, 0.55, 0.25], [0.85, 0.65, 0.45], [0.60, 0.75, 0.35],
]
_DNA_COLOR = [0.85, 0.75, 0.45]   # chromosome = tan


def _solute_atoms(top_path, itp_dir):
    """(labels, color_index, palette_map, n_solute) for non-solvent molecules."""
    natoms_of = {}
    for f in os.listdir(itp_dir):
        if f.endswith(".itp"):
            mt = _moleculetype_name(os.path.join(itp_dir, f))
            if mt:
                natoms_of[mt] = _moleculetype_natoms(os.path.join(itp_dir, f))
    mols, in_mol = [], False
    for ln in open(top_path):
        s = ln.strip()
        if s.startswith("[ molecules ]"):
            in_mol = True
            continue
        if in_mol and s and not s.startswith(";"):
            if s.startswith("["):
                break
            name, count = s.split()[:2]
            mols.append((name, int(count)))

    species_order = []
    for name, _ in mols:
        if name in _SOLVENT:
            continue
        if name not in species_order:
            species_order.append(name)
    palette_map, color_of = {}, {}
    pi = 0
    for sp in species_order:
        if sp == "DNA_SEG":
            palette_map[sp] = _DNA_COLOR
        else:
            palette_map[sp] = _PALETTE[pi % len(_PALETTE)]
            pi += 1
        color_of[sp] = palette_map[sp]

    labels, cidx, palette = [], [], list(palette_map.values())
    pal_index = {sp: i for i, sp in enumerate(palette_map)}
    for name, count in mols:
        if name in _SOLVENT:
            continue
        n = natoms_of.get(name, 0)
        for _ in range(count):
            labels.extend([name] * n)
            cidx.extend([pal_index[name]] * n)
    return labels, np.asarray(cidx, np.uint8), palette_map, len(labels)


def build(dcd, gro, top, itp_dir, out_html, bead_radius_nm=0.30,
          title="Crowded Martini E. coli chunk (cytoplasm + chromosome)",
          background="#05070d"):
    import mdtraj as md
    labels, cidx, palette_map, n_sol = _solute_atoms(top, itp_dir)
    traj = md.load_dcd(dcd, top=gro)
    xyz = traj.xyz[:, :n_sol, :].astype("<f4")   # solute only
    n_frames = xyz.shape[0]
    palette = list(palette_map.values())
    counts = {sp: labels.count(sp) for sp in palette_map}

    frames_b64 = base64.b64encode(np.ascontiguousarray(xyz).tobytes()).decode()
    colors_b64 = base64.b64encode(cidx.tobytes()).decode()
    legend = "".join(
        f'<div class="lrow"><span class="sw" style="background:rgb('
        f'{int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"></span>'
        f'{"chromosome (DNA)" if sp=="DNA_SEG" else sp}'
        f' <span class="n">{counts.get(sp,0):,}</span></div>'
        for sp, c in palette_map.items())

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>html,body{{margin:0;height:100%;overflow:hidden;background:{background};color:#dfe6f3;
font-family:system-ui,sans-serif}}#bar{{position:fixed;top:0;left:0;right:0;padding:10px 16px;z-index:10;
pointer-events:none;background:linear-gradient(180deg,rgba(5,7,13,.92),rgba(5,7,13,0))}}#bar h1{{margin:0;font-size:15px}}
#bar p{{margin:2px 0 0;font-size:12px;opacity:.7}}#legend{{position:fixed;left:12px;bottom:64px;z-index:10;font-size:12px;
background:rgba(5,7,13,.6);padding:8px 10px;border-radius:8px}}.lrow{{display:flex;align-items:center;gap:6px;margin:2px 0}}
.sw{{width:11px;height:11px;border-radius:2px;display:inline-block}}.n{{margin-left:auto;opacity:.6}}
#ctl{{position:fixed;left:0;right:0;bottom:0;z-index:10;display:flex;align-items:center;gap:12px;padding:12px 16px;
background:rgba(5,7,13,.85)}}#ctl button{{font-size:14px;padding:6px 14px;border-radius:6px;border:0;background:#2b3b66;color:#fff;cursor:pointer}}
#frame{{flex:1}}#lab{{font-variant-numeric:tabular-nums;opacity:.8;min-width:120px}}#c{{position:absolute;inset:0}}</style>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script></head><body>
<div id="bar"><h1>{title}</h1><p>{n_sol:,} solute beads (water hidden) &middot; {n_frames} frames &middot; real Martini 3 MD &middot; drag to rotate</p></div>
<div id="legend">{legend}</div><canvas id="c"></canvas>
<div id="ctl"><button id="play">⏸ Pause</button><input id="frame" type="range" min="0" max="{n_frames-1}" value="0" step="1"><span id="lab">frame 0 / {n_frames-1}</span></div>
<script>
const NF={n_frames},NA={n_sol},R={bead_radius_nm},PALETTE={json.dumps(palette)};
function b64(s){{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}}
const xyz=new Float32Array(b64("{frames_b64}").buffer),cidx=b64("{colors_b64}");
const r=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true}});r.setPixelRatio(Math.min(devicePixelRatio,2));r.setSize(innerWidth,innerHeight);
const scene=new THREE.Scene();scene.background=new THREE.Color("{background}");
const cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.01,100000);
const ctr=new THREE.OrbitControls(cam,r.domElement);ctr.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,0.6));const dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(1,1,1);scene.add(dl);
const mesh=new THREE.InstancedMesh(new THREE.SphereGeometry(R,8,6),new THREE.MeshLambertMaterial(),NA);
mesh.instanceColor=new THREE.InstancedBufferAttribute(new Float32Array(NA*3),3);
for(let i=0;i<NA;i++){{const c=PALETTE[cidx[i]]||[.8,.8,.8];mesh.setColorAt(i,new THREE.Color(c[0],c[1],c[2]));}}
mesh.instanceColor.needsUpdate=true;const d=new THREE.Object3D();
function setF(f){{const o=f*NA*3;for(let i=0;i<NA;i++){{d.position.set(xyz[o+i*3],xyz[o+i*3+1],xyz[o+i*3+2]);d.updateMatrix();mesh.setMatrixAt(i,d.matrix);}}mesh.instanceMatrix.needsUpdate=true;}}
setF(0);scene.add(mesh);
const bb=new THREE.Box3(),v=new THREE.Vector3();for(let i=0;i<NA;i++){{v.set(xyz[i*3],xyz[i*3+1],xyz[i*3+2]);bb.expandByPoint(v);}}
const cen=bb.getCenter(new THREE.Vector3()),sz=bb.getSize(new THREE.Vector3()).length();
ctr.target.copy(cen);cam.position.copy(cen).add(new THREE.Vector3(0,0,sz*0.7));cam.updateProjectionMatrix();
let cur=0,play=true,acc=0;const sl=document.getElementById('frame'),lab=document.getElementById('lab'),bt=document.getElementById('play');
bt.onclick=()=>{{play=!play;bt.textContent=play?'⏸ Pause':'▶ Play';}};sl.oninput=()=>{{cur=+sl.value;play=false;bt.textContent='▶ Play';show();}};
function show(){{setF(cur);sl.value=cur;lab.textContent='frame '+cur+' / '+(NF-1);}}
addEventListener("resize",()=>{{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();r.setSize(innerWidth,innerHeight);}});
let last=performance.now();(function loop(t){{requestAnimationFrame(loop);const dt=t-last;last=t;if(play){{acc+=dt;if(acc>180){{acc=0;cur=(cur+1)%NF;show();}}}}ctr.update();r.render(scene,cam);}})(last);
</script></body></html>"""
    open(out_html, "w").write(html)
    return out_html, n_frames, n_sol


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for a in ("dcd", "gro", "top", "itp-dir", "out"):
        ap.add_argument("--" + a, required=True)
    z = ap.parse_args()
    out, nf, ns = build(z.dcd, z.gro, z.top, getattr(z, "itp_dir"), z.out)
    print(f"wrote {out}  ({nf} frames, {ns:,} solute beads, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
