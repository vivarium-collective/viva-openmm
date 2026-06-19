"""Build a self-contained animated 3D viewer of a Martini MD trajectory.

Reads a DCD trajectory + its topology (gro) and the prepared `.top`, then emits
one self-contained HTML that animates every CG bead (three.js instanced spheres,
colored by species) with play/pause + a frame slider. Open it in any browser; it
embeds all frames inline, so the single file is shareable.

Usage::

    python scripts/make_traj_viewer.py \
        --dcd /tmp/slice_small/traj.dcd \
        --gro /tmp/slice_small/relaxed.gro \
        --top /tmp/slice_small/_prepared.top \
        --itp-dir /tmp/slice_small/templates \
        --out /tmp/slice_small/md_viewer.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os

import numpy as np

from pbg_openmm.topology import _moleculetype_name, _moleculetype_natoms

# Distinct, readable palette assigned per species (cycled if >len).
_PALETTE = [
    [0.95, 0.55, 0.25], [0.45, 0.80, 0.50], [0.35, 0.60, 0.95],
    [0.90, 0.40, 0.55], [0.95, 0.85, 0.30], [0.70, 0.55, 0.90],
    [0.40, 0.85, 0.85], [0.85, 0.65, 0.45],
]


def _per_atom_species(top_path, itp_dir):
    """Walk the prepared .top [molecules] and return (labels, color_index, palette_map).

    labels: per-atom species name; color_index: per-atom int into the palette.
    """
    # moltype name -> natoms, and moltype -> slug (filename stem)
    mol_natoms, mol_slug = {}, {}
    for f in os.listdir(itp_dir):
        if not f.endswith(".itp"):
            continue
        p = os.path.join(itp_dir, f)
        name = _moleculetype_name(p)
        if name:
            mol_natoms[name] = _moleculetype_natoms(p)
            mol_slug[name] = f[:-4]
    # read [molecules] order
    mols, in_mol = [], False
    for ln in open(top_path):
        s = ln.strip()
        if s.startswith("[ molecules ]"):
            in_mol = True
            continue
        if in_mol and s and not s.startswith(";"):
            if s.startswith("["):
                break
            parts = s.split()
            mols.append((parts[0], int(parts[1])))
    species_order = []
    for moltype, _ in mols:
        slug = mol_slug.get(moltype, moltype)
        if slug not in species_order:
            species_order.append(slug)
    color_of = {sp: i % len(_PALETTE) for i, sp in enumerate(species_order)}

    labels, color_index = [], []
    for moltype, count in mols:
        slug = mol_slug.get(moltype, moltype)
        n = mol_natoms.get(moltype, 0)
        for _ in range(count):
            labels.extend([slug] * n)
            color_index.extend([color_of[slug]] * n)
    palette_map = {sp: _PALETTE[color_of[sp]] for sp in species_order}
    return labels, np.asarray(color_index, dtype=np.uint8), palette_map


def build_traj_viewer(dcd, gro, top, itp_dir, out_html,
                      title="Martini MD trajectory (parsimony E. coli slice)",
                      bead_radius_nm=0.26, background="#05070d"):
    import mdtraj as md
    traj = md.load_dcd(dcd, top=gro)
    xyz = traj.xyz.astype("<f4")                  # (frames, atoms, 3) in nm
    n_frames, n_atoms, _ = xyz.shape

    labels, color_index, palette_map = _per_atom_species(top, itp_dir)
    if len(color_index) != n_atoms:               # fallback: single color
        color_index = np.zeros(n_atoms, dtype=np.uint8)
        palette_map = {"all beads": _PALETTE[0]}

    frames_b64 = base64.b64encode(xyz.tobytes()).decode("ascii")
    colors_b64 = base64.b64encode(color_index.tobytes()).decode("ascii")
    palette = [_PALETTE[i] for i in range(len(_PALETTE))]

    counts = {sp: int(np.sum(np.asarray(labels) == sp)) for sp in palette_map}
    legend = "".join(
        f'<div class="lrow"><span class="sw" style="background:rgb('
        f'{int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"></span>{sp}'
        f' <span class="n">{counts.get(sp,0):,}</span></div>'
        for sp, c in palette_map.items())

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  html,body{{margin:0;height:100%;overflow:hidden;background:{background};color:#dfe6f3;
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
  #bar{{position:fixed;top:0;left:0;right:0;padding:10px 16px;z-index:10;pointer-events:none;
    background:linear-gradient(180deg,rgba(5,7,13,.92),rgba(5,7,13,0))}}
  #bar h1{{margin:0;font-size:15px;font-weight:600}} #bar p{{margin:2px 0 0;font-size:12px;opacity:.7}}
  #legend{{position:fixed;left:12px;bottom:64px;z-index:10;font-size:12px;
    background:rgba(5,7,13,.6);padding:8px 10px;border-radius:8px}}
  .lrow{{display:flex;align-items:center;gap:6px;margin:2px 0}}
  .sw{{width:11px;height:11px;border-radius:2px;display:inline-block}} .n{{margin-left:auto;opacity:.6}}
  #ctl{{position:fixed;left:0;right:0;bottom:0;z-index:10;display:flex;align-items:center;gap:12px;
    padding:12px 16px;background:rgba(5,7,13,.85)}}
  #ctl button{{font-size:14px;padding:6px 14px;border-radius:6px;border:0;background:#2b3b66;color:#fff;cursor:pointer}}
  #frame{{flex:1}} #lab{{font-variant-numeric:tabular-nums;opacity:.8;min-width:120px}}
  #c{{position:absolute;inset:0}}
</style>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head><body>
  <div id="bar"><h1>{title}</h1>
    <p>{n_atoms:,} Martini CG beads &middot; {n_frames} frames &middot; real OpenMM Martini 3 MD &middot; drag to rotate</p></div>
  <div id="legend">{legend}</div>
  <canvas id="c"></canvas>
  <div id="ctl">
    <button id="play">⏸ Pause</button>
    <input id="frame" type="range" min="0" max="{n_frames-1}" value="0" step="1">
    <span id="lab">frame 0 / {n_frames-1}</span>
  </div>
  <script>
    const NF={n_frames}, NA={n_atoms}, R={bead_radius_nm};
    const PALETTE={json.dumps(palette)};
    function b64(s){{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u;}}
    const xyz=new Float32Array(b64("{frames_b64}").buffer);   // NF*NA*3
    const cidx=b64("{colors_b64}");                            // NA
    const renderer=new THREE.WebGLRenderer({{canvas:document.getElementById('c'),antialias:true}});
    renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(innerWidth,innerHeight);
    const scene=new THREE.Scene(); scene.background=new THREE.Color("{background}");
    const camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.01,100000);
    const controls=new THREE.OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
    scene.add(new THREE.AmbientLight(0xffffff,0.6));
    const dl=new THREE.DirectionalLight(0xffffff,0.8); dl.position.set(1,1,1); scene.add(dl);
    const geo=new THREE.SphereGeometry(R,8,6);
    const mat=new THREE.MeshLambertMaterial({{vertexColors:false}});
    const mesh=new THREE.InstancedMesh(geo,mat,NA);
    mesh.instanceColor=new THREE.InstancedBufferAttribute(new Float32Array(NA*3),3);
    for(let i=0;i<NA;i++){{const c=PALETTE[cidx[i]]||[0.8,0.8,0.8];mesh.setColorAt(i,new THREE.Color(c[0],c[1],c[2]));}}
    mesh.instanceColor.needsUpdate=true;
    const dummy=new THREE.Object3D();
    function setFrame(f){{
      const off=f*NA*3;
      for(let i=0;i<NA;i++){{dummy.position.set(xyz[off+i*3],xyz[off+i*3+1],xyz[off+i*3+2]);dummy.updateMatrix();mesh.setMatrixAt(i,dummy.matrix);}}
      mesh.instanceMatrix.needsUpdate=true;
    }}
    setFrame(0); scene.add(mesh);
    // frame camera on bounding box of frame 0
    const box=new THREE.Box3(); const v=new THREE.Vector3();
    for(let i=0;i<NA;i++){{v.set(xyz[i*3],xyz[i*3+1],xyz[i*3+2]);box.expandByPoint(v);}}
    const ctr=box.getCenter(new THREE.Vector3()), sz=box.getSize(new THREE.Vector3()).length();
    controls.target.copy(ctr); camera.position.copy(ctr).add(new THREE.Vector3(0,0,sz*0.7)); camera.updateProjectionMatrix();
    // playback
    let cur=0, playing=true, acc=0; const slider=document.getElementById('frame'), lab=document.getElementById('lab');
    const btn=document.getElementById('play');
    btn.onclick=()=>{{playing=!playing;btn.textContent=playing?'⏸ Pause':'▶ Play';}};
    slider.oninput=()=>{{cur=+slider.value;playing=false;btn.textContent='▶ Play';show();}};
    function show(){{setFrame(cur);slider.value=cur;lab.textContent='frame '+cur+' / '+(NF-1);}}
    addEventListener("resize",()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);}});
    let last=performance.now();
    (function loop(t){{requestAnimationFrame(loop);const dt=t-last;last=t;
      if(playing){{acc+=dt;if(acc>180){{acc=0;cur=(cur+1)%NF;show();}}}}
      controls.update();renderer.render(scene,camera);}})(last);
  </script>
</body></html>
"""
    with open(out_html, "w") as fh:
        fh.write(html)
    return out_html, n_frames, n_atoms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dcd", required=True)
    ap.add_argument("--gro", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--itp-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, nf, na = build_traj_viewer(a.dcd, a.gro, a.top, a.itp_dir, a.out)
    print(f"wrote {out}  ({nf} frames, {na:,} beads, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
