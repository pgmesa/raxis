"""
The animated header of the README: a pixel-art brick oven whose wall forms the
word "Raxis".

The letters are holes in the brick wall. The animation tells the story of the
project in four acts, then loops on the last two:
  1. build:  the stone base, then the walls course by course from the bottom,
             then the chimney; the letter holes show the dark, cold oven;
  2. fill:   the letters fill with fire from the bottom row up (the oven heats);
  3. sweep:  an "operation" walks down each column, column by column (one axis
             at a time, as x.sum(dim=0) visits a tensor);
  4. fire:   flames climb inside the letters, sparks escape from them and smoke
             leaves the chimney; then calm until the next sweep.

Everything is drawn as 1x1 rects (pixels) animated with CSS, since a README
image cannot run JavaScript. Readers with prefers-reduced-motion see the
finished oven, still.

Usage:
    python scripts/make_readme_header.py [--out docs/assets/raxis-header.svg]
"""

import argparse
import math
import random
from pathlib import Path


# =============================================================================
# Drawing
# =============================================================================

# A bold pixel font: 8 rows, 2-pixel strokes, so every letter is a real window
# in the wall. Lowercase letters start on row 2.
GLYPHS = {
    'R': ['XXXXXX.', 'XXXXXXX', 'XX...XX', 'XXXXXXX', 'XXXXXX.', 'XX..XX.', 'XX...XX', 'XX...XX'],
    'a': ['.......', '.......', 'XXXXXX.', '.....XX', '.XXXXXX', 'XX...XX', 'XX...XX', '.XXXXXX'],
    'x': ['.......', '.......', 'XX...XX', '.XX.XX.', '..XXX..', '..XXX..', '.XX.XX.', 'XX...XX'],
    'i': ['XX', 'XX', '..', 'XX', 'XX', 'XX', 'XX', 'XX'],
    's': ['......', '......', '.XXXXX', 'XX....', 'XXXXX.', '.XXXXX', '....XX', 'XXXXX.'],
}
WORD = 'Raxis'
LETTER_ROWS = 8
LETTER_GAP = 2

# Fire, from the top rows of the letters to the bottom ones (Endesga 32 tones)
FIRE = ['#fee761', '#feae34', '#f77622']
FLASH, HOT = '#fee761', '#ffffff'
OVEN_INSIDE = '#1a1215'

BRICKS = ['#8f4a37', '#9c533d', '#83432f']
MORTAR = '#4a2a24'               # also the shadow of the wall's thickness around the holes
CHIMNEY = ['#7a3f30', '#8a4735']
CAP = '#4a4146'
STONES = ['#9c8670', '#8b7560', '#a8927a']
STONE_TOP, STONE_JOINT = '#bba58c', '#5e4a3c'
WOOD_STORE, BARK, LOG_CORE = '#2a1f1c', '#5a3a24', '#c79a6a'
SMOKE = ['#9a9096', '#7d747a']

# Geometry, in pixels
WALL_SIDE = 6        # wall at each side of the word
WORD_TOP = 12        # first row of the letters
BODY_TOP = 5         # first row of the oven body (the vault springs from here)
VAULT_RISE = 5       # rows the vault rises from its sides to its middle
WALL_BELOW = 3       # rows of wall under the letters
BASE_ROWS = 6        # the stone base, with the wood store in it
CHIMNEY_ABOVE = 2    # rows the chimney rises above the top of the body
SMOKE_RISE = 4       # rows a puff of smoke climbs before it fades

# Time, in seconds
CYCLE = 12.0                     # sweep + fire + calm, repeated forever
FILL_STEP = 0.008                # between two letter pixels filling
SWEEP_STEP, SWEEP_FLASH = 0.018, 0.04
FIRE_FOR = 2.4
PULSE, ROW_LAG = 0.26, 0.04      # flame pulse period, and how much later a pulse starts one row up
SPARKS, SPARK_RISE = 65, 6
TONGUE_PROBABILITY = 0.45        # columns with a 1-pixel tongue of flame above the letter

SCALE = 12                       # display pixels per drawing pixel


def letter_pixels():
    """(row, col) of every pixel of the word, and the word's width."""
    pixels, col = [], 0
    for ch in WORD:
        glyph = GLYPHS[ch]
        pixels += [(r, col + c) for r, line in enumerate(glyph) for c, v in enumerate(line) if v == 'X']
        col += len(glyph[0]) + LETTER_GAP
    return pixels, col - LETTER_GAP


def draw_oven(width, holes, rng):
    """Color of every pixel of the oven, and where its parts are."""
    w = width + 2 * WALL_SIDE
    body_bottom = WORD_TOP + LETTER_ROWS + WALL_BELOW
    h = body_bottom + BASE_ROWS
    grid, base = {}, set()

    # the body: brick courses 3 rows high (2 of brick, 1 of mortar), bricks 5
    # wide, every other course shifted half a brick (running bond); a vault on top
    brick_color = {}
    for y in range(BODY_TOP, body_bottom):
        for x in range(1, w - 1):
            dx = (x + 0.5 - w / 2) / ((w - 2) / 2)
            if y + 0.5 < BODY_TOP + VAULT_RISE * (1 - math.sqrt(max(0.0, 1 - dx * dx))):
                continue
            course, row_in_course = divmod(y, 3)
            brick, col_in_brick = divmod(x + (3 if course % 2 else 0), 6)
            if row_in_course == 2 or col_in_brick == 5:
                grid[x, y] = MORTAR
            else:
                if (course, brick) not in brick_color:
                    brick_color[course, brick] = rng.choice(BRICKS)
                grid[x, y] = brick_color[course, brick]

    # the letters are holes; the bricks just above and left of a hole are in
    # the shadow of the wall's thickness
    for p in holes:
        grid.pop(p, None)
    for (x, y) in list(grid):
        if (x, y + 1) in holes or (x + 1, y) in holes:
            grid[x, y] = MORTAR

    # the base: stones of uneven width on 2-row courses, a lit top row, and an
    # arched wood store in the middle with logs seen end on
    store_x0, store_x1 = w // 2 - 6, w // 2 + 6
    store_cx, store_rx = (store_x0 + store_x1) / 2, (store_x1 - store_x0) / 2
    stone_widths = {}
    for by in range(BASE_ROWS):
        y = body_bottom + by
        for x in range(w):
            base.add((x, y))
            in_store = (store_x0 <= x < store_x1 and by >= 1 and
                        (by >= 3 or ((x + 0.5 - store_cx) / store_rx) ** 2 + ((by + 0.5 - 3) / 2) ** 2 <= 1))
            if in_store:
                lx = x - store_x0
                bottom_logs = by >= BASE_ROWS - 2 and 1 <= lx <= 10
                top_logs = by == BASE_ROWS - 3 and 2 <= lx <= 9
                if bottom_logs or top_logs:
                    cut = lx % 3 == 1 and by in (BASE_ROWS - 1, BASE_ROWS - 3)
                    grid[x, y] = LOG_CORE if cut else BARK
                else:
                    grid[x, y] = WOOD_STORE
                continue
            if by == 0:
                grid[x, y] = STONE_TOP
                continue
            row = (by - 1) // 2
            if row not in stone_widths:
                stone_widths[row] = [rng.choice((3, 4, 5, 6)) for _ in range(w)]
            edge, k = -(row % 2) * 2, 0
            while edge + stone_widths[row][k] <= x:
                edge += stone_widths[row][k]
                k += 1
            last_col = x == edge + stone_widths[row][k] - 1
            grid[x, y] = STONE_JOINT if last_col else STONES[(row * 7 + k) % 3]

    # the chimney, on the right, with a cap one pixel wider at each side
    chimney_x0, chimney_x1 = w - 12, w - 7
    cap_y = BODY_TOP - CHIMNEY_ABOVE
    chimney = set()
    for y in range(cap_y, BODY_TOP + 3):
        for x in range(chimney_x0 - 1, chimney_x1 + 1):
            if chimney_x0 <= x < chimney_x1:
                grid[x, y] = CAP if y == cap_y else (MORTAR if y % 3 == 2 else CHIMNEY[(x + y) % 2])
            elif y == cap_y:
                grid[x, y] = CAP
            else:
                continue
            chimney.add((x, y))
    return grid, base, chimney, (chimney_x0, chimney_x1, cap_y), w, h


# =============================================================================
# Animation
# =============================================================================

def build_svg(seed=11):
    rng = random.Random(seed)
    letters, width = letter_pixels()
    ox, oy = WALL_SIDE, WORD_TOP
    holes = {(ox + c, oy + r) for r, c in letters}
    grid, base, chimney, (chimney_x0, chimney_x1, cap_y), w, h = draw_oven(width, holes, rng)

    # --- build: base, walls course by course from the bottom, chimney ---------
    delay = {p: 0.35 for p in base}
    t = 0.65
    wall = [p for p in grid if p not in base and p not in chimney]
    courses = sorted({y // 3 for _, y in wall} | {y // 3 for _, y in holes}, reverse=True)
    lay = lambda x, y: t + courses.index(y // 3) * 0.13 + x * 0.006
    for (x, y) in wall:
        delay[x, y] = lay(x, y)
    hole_delay = {(x, y): lay(x, y) for (x, y) in holes}   # the cold oven shows as each course is laid
    t += len(courses) * 0.13 + 0.35
    chimney_rows = sorted({y for _, y in chimney}, reverse=True)
    for (x, y) in chimney:
        delay[x, y] = t + chimney_rows.index(y) * 0.07
    t += len(chimney_rows) * 0.07 + 0.45

    # --- fill: bottom row first (the temperature rising), left to right -------
    fill_order = sorted(letters, key=lambda p: (-p[0], p[1]))
    fill_at = t
    first_cycle = fill_at + len(fill_order) * FILL_STEP + 0.8

    # --- cycles: sweep, then fire ---------------------------------------------
    sweep_order = {p: i for i, p in enumerate(sorted(letters, key=lambda p: (p[1], p[0])))}
    fire_at = len(letters) * SWEEP_STEP + 0.3       # the fire starts when the sweep is done
    fire_end = fire_at + FIRE_FOR
    pct = lambda s: f'{100 * s / CYCLE:.2f}%'
    pulses, tt = [], fire_at
    while tt < fire_end - 0.3:
        pulses += [f'{pct(tt)} {{ opacity: 0; }}', f'{pct(tt + PULSE * 0.3)} {{ opacity: .95; }}',
                   f'{pct(tt + PULSE * 0.7)} {{ opacity: 0; }}']
        tt += PULSE
    css = f'''<style>
  .b, .p, .e, .s, .k {{ opacity: 0; }}
  .b {{ animation: on 0.12s linear forwards; }}
  .p {{ animation: on 0.1s linear forwards, sweep {CYCLE}s steps(1) infinite; }}
  @keyframes on {{ to {{ opacity: 1; }} }}
  @keyframes sweep {{ 0% {{ fill: {HOT}; }} {pct(SWEEP_FLASH)} {{ fill: var(--c); }} }}
  @keyframes flame {{ 0% {{ opacity: 0; }} {' '.join(pulses)} 100% {{ opacity: 0; }} }}
  @keyframes spark {{ 0% {{ opacity: 0; transform: translateY(0); }} 0.3% {{ opacity: 1; }}
    2.4% {{ opacity: .85; fill: {FIRE[1]}; }} {pct(0.36)} {{ opacity: 0; transform: translateY(-{SPARK_RISE}px); fill: {FIRE[2]}; }} 100% {{ opacity: 0; }} }}
  @keyframes smoke {{ 0% {{ opacity: 0; transform: translate(0, 0); }} 1% {{ opacity: .75; }}
    {pct(1.4)} {{ opacity: 0; transform: translate(var(--dx), -{SMOKE_RISE}px); }} 100% {{ opacity: 0; }} }}
  @media (prefers-reduced-motion: reduce) {{ .b, .p {{ animation: none; opacity: 1; }} .e, .s, .k {{ animation: none; opacity: 0; }} }}
</style>'''
    # Fades are linear, not steps(1): with thousands of short steps(1)
    # animations Chromium dropped the final state of some, leaving whole
    # columns of bricks hidden. 0.1 s still reads as a pixel popping in.

    def rect(x, y, color, cls, style):
        return f'<rect class="{cls}" x="{x}" y="{y}" width="1" height="1" fill="{color}" style="{style}"/>'

    def fire_band(r):
        return FIRE[0] if r <= 2 else FIRE[1] if r <= 5 else FIRE[2]

    out = [rect(x, y, c, 'b', f'animation-delay:{delay[x, y]:.3f}s') for (x, y), c in grid.items()]
    out += [rect(x, y, OVEN_INSIDE, 'b', f'animation-delay:{d:.3f}s') for (x, y), d in hole_delay.items()]
    for i, (r, c) in enumerate(fill_order):
        color = fire_band(r)
        out.append(rect(ox + c, oy + r, color, 'p', f'--c:{color};animation-delay:{fill_at + i * FILL_STEP:.3f}s,'
                                                    f'{first_cycle + sweep_order[r, c] * SWEEP_STEP:.3f}s'))
    # flames inside the letters: pulses that start at the bottom row and climb
    for (r, c) in fill_order:
        color = HOT if r < 4 and rng.random() < 0.25 else FLASH
        d = first_cycle + (LETTER_ROWS - 1 - r) * ROW_LAG + rng.uniform(0, 0.08)
        out.append(rect(ox + c, oy + r, color, 'e', f'animation:flame {CYCLE}s steps(1) {d:.3f}s infinite'))
    tops = {}
    for r, c in letters:
        tops[c] = min(tops.get(c, LETTER_ROWS), r)
    # tongues of flame licking one pixel above some columns of the letters
    for c, top in sorted(tops.items()):
        if rng.random() < TONGUE_PROBABILITY:
            d = first_cycle + (LETTER_ROWS - top) * ROW_LAG + rng.uniform(0, 0.1)
            out.append(rect(ox + c, oy + top - 1, FLASH, 'e', f'animation:flame {CYCLE}s steps(1) {d:.3f}s infinite'))
    # sparks escaping from the letters
    columns = sorted(tops)
    for _ in range(SPARKS):
        c = rng.choice(columns)
        d = first_cycle + fire_at + rng.uniform(0, FIRE_FOR - 0.4)
        out.append(rect(ox + c, oy + tops[c] - 1, FLASH, 's', f'animation:spark {CYCLE}s steps(6) {d:.3f}s infinite'))
    # puffs of smoke from the chimney, during the fire and a little after
    puffs = []
    for i in range(9):
        x = rng.randint(chimney_x0, chimney_x1 - 2)
        d = first_cycle + fire_at + i * (FIRE_FOR + 1.0) / 9
        dx = rng.choice((-2, -1, 1, 2))
        puffs.append(f'<rect class="k" x="{x}" y="{cap_y - 2}" width="2" height="2" fill="{SMOKE[i % 2]}" '
                     f'style="--dx:{dx}px;animation:smoke {CYCLE}s steps(7) {d:.3f}s infinite"/>')

    # room above the oven only for the smoke: its highest point is the top row
    shift = max(0, SMOKE_RISE - (cap_y - 2))
    height = h + shift
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {height}" shape-rendering="crispEdges" '
            f'width="{w * SCALE}" height="{height * SCALE}" role="img" aria-label="Raxis">{css}'
            f'<g transform="translate(0 {shift})">{"".join(puffs)}{"".join(out)}</g></svg>\n')


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, default=root / 'docs' / 'assets' / 'raxis-header.svg')
    args = parser.parse_args()
    args.out.write_text(build_svg(), encoding='ascii')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
