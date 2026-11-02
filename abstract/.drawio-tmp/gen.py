"""Write the Fig. 2-4 YAML specs with one shared style vocabulary (same as Fig. 1)."""
import sys
STY = {
  'pro': ('#EAF2FB', '#438DD5'),   # processing on the PC
  'cap': ('#D6E4F2', '#08427B'),   # phone / browser side
  'art': ('#EDEDED', '#686868'),   # data artifacts
  'gate': ('#F6F0DA', '#9C8A2E'),  # decisions / gates
}
HEAD = '''meta:
  profile: academic-paper
  figureType: {ft}
  theme: academic-color
  palette: c4-blue
  layout: horizontal
  font:
    primary: "Helvetica, Arial, sans-serif"
    cjk: "Helvetica, Arial, sans-serif"
    formula: "Helvetica, Arial, sans-serif"
  title: "{title}"
  description: "{desc}"
  legend: "{legend}"

nodes:
'''
def node(i, label, typ, kind, x, y, w=160, h=56):
    f, s = STY[kind]
    return (f'  - id: {i}\n    label: "{label}"\n    type: {typ}\n    bounds: {{ x: {x}, y: {y}, width: {w}, height: {h} }}\n'
            f'    style: {{ fillColor: "{f}", strokeColor: "{s}", fontColor: "#1A1A1A", fontSize: 15 }}\n')
def edge(a, b, label=None, typ='primary', wp=None):
    l = f', label: "{label}"' if label else ''
    w = ', waypoints: [' + ', '.join(f'{{ x: {x}, y: {y} }}' for x, y in wp) + ']' if wp else ''
    return f'  - {{ from: {a}, to: {b}, type: {typ}{l}{w}, style: {{ fontSize: 13, fontColor: "#333333" }} }}\n'
def write(path, ft, title, desc, legend, nodes, edges):
    open(path, 'w').write(HEAD.format(ft=ft, title=title, desc=desc, legend=legend) + ''.join(nodes) + '\nedges:\n' + ''.join(edges))

L = 'Light blue: processing steps. Grey: data. Sand diamonds: automatic gates.'
write('abstract/.drawio-tmp/fig2-pipeline/fig2-pipeline.yaml', 'workflow', 'Reconstruction pipeline',
  'Five resumable steps of wayfarer.py with the artifacts passed between them.', L, [
  node('video', 'Phone video(s)\\nor photo folder', 'document', 'art', 20, 108, 150, 60),
  node('frames', '1 frames\\nffmpeg + Laplacian', 'process', 'pro', 240, 110),
  node('sfm', '2 sfm\\nCOLMAP 4.2 global', 'process', 'pro', 520, 110),
  node('gate', '≥ 50 %\\nregistered?', 'decision', 'gate', 760, 98, 130, 80),
  node('stop', 'stop with hint', 'terminal', 'art', 760, 10, 130, 40),
  node('train', '3 train\\nsplatfacto (gsplat)', 'process', 'pro', 960, 110),
  node('export', '4 export\\nns-export + ns-eval', 'process', 'pro', 960, 260),
  node('navgrid', '5 navgrid\\nnavgrid.py', 'process', 'pro', 640, 260),
  node('post', 'post-steps: autopoi,\\nsurfaces, floorplan', 'process', 'pro', 640, 390, 160, 56),
  node('pkg', 'Scene package\\nscene.json + scene.sog', 'database', 'art', 260, 244, 200, 94),
 ], [
  edge('video', 'frames', 'clips'),
  edge('frames', 'sfm', '≤ 500 sharp JPEGs'),
  edge('sfm', 'gate', 'sparse/0'),
  edge('gate', 'stop', 'no'),
  edge('gate', 'train', 'yes'),
  edge('train', 'export', 'checkpoint'),
  edge('export', 'navgrid', 'splat.ply, metrics'),
  edge('navgrid', 'pkg', 'grid, SOG, POIs'),
  edge('navgrid', 'post', 'scene dir'),
  edge('post', 'pkg', 'POIs, textures, SVG', 'data'),
 ])

write('abstract/.drawio-tmp/fig3-sop/fig3-sop.yaml', 'workflow', 'Standard operating procedure',
  'Operator and PC steps from installation to the published twin, with the two retake loops.',
  'Dark blue: done by the operator with the phone or browser. Light blue: done on the PC. Sand diamonds: checks.', [
  node('s1', '1 Install\\ndocs/SETUP.md', 'process', 'pro', 20, 40),
  node('s2', '2 Prepare space\\nlights, doors, marker', 'process', 'cap', 220, 40),
  node('s3', '3 Phone settings\\n1x lens, AE/AF lock', 'process', 'cap', 420, 40),
  node('s4', '4 Film\\nSOP capture path', 'process', 'cap', 620, 40),
  node('s5', '5 Pre-flight\\npreflight.py', 'process', 'pro', 820, 40),
  node('d1', 'Take\\npasses?', 'decision', 'gate', 835, 170, 130, 80),
  node('r1', 'Film short\\nretake clips', 'process', 'cap', 1060, 182, 140, 56),
  node('s6', '6 Build twin\\nStudio or CLI', 'process', 'pro', 620, 182),
  node('d2', '7 Gates\\nOK?', 'decision', 'gate', 435, 170, 130, 80),
  node('r2', 'Re-film marker /\\nre-measure ceiling', 'process', 'cap', 420, 310),
  node('s8', '8 Open viewer\\n?scene=name', 'terminal', 'cap', 200, 182, 160, 56),
 ], [
  edge('s1', 's2'), edge('s2', 's3'), edge('s3', 's4'), edge('s4', 's5', 'clips'),
  edge('s5', 'd1', 'report'),
  edge('d1', 'r1', 'no'), edge('r1', 's5', 'add clips', 'primary', [(1130, 68)]),
  edge('d1', 's6', 'yes'),
  edge('s6', 'd2', 'log'),
  edge('d2', 'r2', 'no'),
  edge('d2', 's8', 'yes'),
 ])

write('abstract/.drawio-tmp/fig4-navgrid/fig4-navgrid.yaml', 'workflow', 'Navigation-grid builder',
  'navgrid.py turns an unscaled splat into a level, metric occupancy grid.',
  'Light blue: navgrid.py stages. Grey: inputs and outputs.', [
  node('splat', 'splat.ply +\\ncamera poses', 'document', 'art', 20, 128, 150, 60),
  node('filter', 'Filter floaters\\nopacity < 0.3', 'process', 'pro', 220, 130),
  node('floor', 'Floor RANSAC\\nlevel to +Y, y = 0', 'process', 'pro', 420, 130),
  node('scale', 'Metric scale\\npriority + 12 % check', 'process', 'pro', 610, 130, 180, 56),
  node('refs', 'ArUco marker, ceiling H,\\ncamera height 1.4 m', 'document', 'art', 610, 6, 180, 60),
  node('yaw', 'Manhattan yaw\\nwalls on axes', 'process', 'pro', 880, 130),
  node('band', 'Obstacles 0.12-1.9 m\\n+ depth carving', 'process', 'pro', 880, 260),
  node('walk', 'Walkable floor\\nlinked to camera path', 'process', 'pro', 590, 260, 180, 56),
  node('out', 'scene.json grid\\n+ scene.sog', 'database', 'art', 290, 243, 190, 90),
 ], [
  edge('splat', 'filter'), edge('filter', 'floor'), edge('floor', 'scale'),
  edge('refs', 'scale', 'scale references', 'data'),
  edge('scale', 'yaw', 'metres'), edge('yaw', 'band'), edge('band', 'walk', '10 cm cells'), edge('walk', 'out'),
 ])
