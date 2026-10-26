"""Every export of raxis.autograd.graph for one small model, side by side.

    python examples/autograd_graph.py [out_dir]

Writes into out_dir (default: autograd_graph_out next to this script) and
opens index.html:
  <case>.dot / .mmd    text: to_dot() and to_mermaid(), no dependencies
  <case>.svg           image: render(), needs pip install 'raxis[viz]' and
                       the Graphviz binaries (the `dot` command) on the PATH
  index.html           both drawings of each case; mermaid.js draws Mermaid
"""
import html
import sys
import webbrowser
from pathlib import Path

import raxis
from raxis import nn
from raxis.autograd import graph


out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "autograd_graph_out"
out_dir = out_dir.absolute()
out_dir.mkdir(parents=True, exist_ok=True)

raxis.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
params = dict(model.named_parameters())
x = raxis.randn((5, 4))
y = raxis.randint(0, 3, (5,))
loss = nn.functional.cross_entropy(model(x), y)


def export(name, title, view):
    """Write the text exports and the image of the graph as it is now."""
    case = {
        "title": title,
        "dot": graph.to_dot(loss, params, view=view),
        "mermaid": graph.to_mermaid(loss, params, view=view),
        "svg": None,
    }
    (out_dir / f"{name}.dot").write_text(case["dot"])
    (out_dir / f"{name}.mmd").write_text(case["mermaid"])
    try:
        case["svg"] = graph.render(loss, out_dir / f"{name}.svg", params, view=view).read_text()
    except (ImportError, RuntimeError) as e:
        print(f"{name}: no image ({e})")
    return case


# Saved tensors are freed by backward(), so the first two graphs are exported
# before it and the third one after
cases = [
    export("backward", "view='backward': autograd nodes and the tensors each one saved", "backward"),
    export("forward", "view='forward': the same graph with the edges reversed, forward op names", "forward"),
]
loss.backward()
cases.append(export("after_backward", "After backward(): the saved tensors are released", "backward"))

sections = []
for case in cases:
    image = case["svg"] or "<p>No Graphviz: install it to see render()'s output.</p>"
    sections.append(f"""
<section>
  <h2>{html.escape(case["title"])}</h2>
  <div class="row">
    <div class="col"><h3>render() (Graphviz)</h3>{image}</div>
    <div class="col"><h3>to_mermaid() (mermaid.js)</h3><pre class="mermaid">{html.escape(case["mermaid"])}</pre></div>
  </div>
  <details><summary>to_dot() text</summary><pre>{html.escape(case["dot"])}</pre></details>
  <details><summary>to_mermaid() text</summary><pre>{html.escape(case["mermaid"])}</pre></details>
</section>""")

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>raxis autograd graph</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; }}
  .row {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .col {{ flex: 1 1 420px; background: white; border: 1px solid #ddd; padding: 12px; overflow: auto; }}
  .col svg {{ max-width: 100%; height: auto; }}
  pre {{ background: #f0f0f0; padding: 8px; overflow: auto; font-size: 12px; }}
  pre.mermaid {{ background: white; }}
  section {{ margin-bottom: 40px; }}
</style>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{ startOnLoad: true }});
</script>
</head><body>
<h1>raxis.autograd.graph</h1>
<p>Linear(4, 8) -> ReLU -> Linear(8, 3) and cross_entropy on a batch of 5.
Leaves are named from <code>dict(model.named_parameters())</code>.</p>
{''.join(sections)}
</body></html>
"""
index = out_dir / "index.html"
index.write_text(page, encoding="utf-8")
print("wrote", index)
webbrowser.open(index.as_uri())
