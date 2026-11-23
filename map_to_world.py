"""Turn a nav2 map (room.yaml + room.pgm) into a Gazebo Classic world with the walls extruded.

usage: python3 map_to_world.py room.yaml        -> writes room.sdf (world name "room")
"""
import argparse
import os

WALL_HEIGHT = 0.5  # m, only needs to be taller than the lidar plane


def read_yaml(path: str) -> dict:
    # ponytail: flat "key: value" parse, nav2 map yamls are flat; use PyYAML if that changes
    out = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                out[key.strip()] = value.strip()
    return out


def read_pgm(path: str) -> tuple[int, int, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    tokens, i = [], 0
    while len(tokens) < 4:  # magic, width, height, maxval
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            i = data.index(b"\n", i)
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    if tokens[0] != b"P5" or int(tokens[3]) > 255:
        raise ValueError(f"{path}: expected an 8-bit binary PGM (P5)")
    width, height = int(tokens[1]), int(tokens[2])
    return width, height, data[i + 1:i + 1 + width * height]


def rectangles(occ: list[list[bool]]) -> list[tuple[int, int, int, int]]:
    """Merge occupied cells into (row, col, n_rows, n_cols) boxes: runs along each row, stacked when identical."""
    open_runs, boxes = {}, []  # open_runs: (c0, c1) -> [first_row, n_rows]
    for r, row in enumerate(occ):
        runs, c = set(), 0
        while c < len(row):
            if row[c]:
                c0 = c
                while c < len(row) and row[c]:
                    c += 1
                runs.add((c0, c))
            else:
                c += 1
        for key in list(open_runs):
            if key in runs:
                open_runs[key][1] += 1
            else:
                r0, n = open_runs.pop(key)
                boxes.append((r0, key[0], n, key[1] - key[0]))
        for key in runs - open_runs.keys():
            open_runs[key] = [r, 1]
    boxes += [(r0, c0, n, c1 - c0) for (c0, c1), (r0, n) in open_runs.items()]
    return boxes


def box_sdf(i: int, x: float, y: float, sx: float, sy: float) -> str:
    geom = f"<geometry><box><size>{sx:.3f} {sy:.3f} {WALL_HEIGHT}</size></box></geometry>"
    return (f'<collision name="c{i}"><pose>{x:.3f} {y:.3f} {WALL_HEIGHT / 2} 0 0 0</pose>{geom}</collision>'
            f'<visual name="v{i}"><pose>{x:.3f} {y:.3f} {WALL_HEIGHT / 2} 0 0 0</pose>{geom}</visual>\n')


def world_sdf(name: str, walls: str) -> str:
    # Gazebo Classic 11 (ROS Humble + TurtleBot3)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
<world name="{name}">
<include><uri>model://sun</uri></include>
<include><uri>model://ground_plane</uri></include>
<model name="walls"><static>true</static><link name="link">
{walls}</link></model>
</world>
</sdf>
"""


def convert(yaml_path: str, out_path: str) -> int:
    meta = read_yaml(yaml_path)
    pgm_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), meta["image"])
    width, height, pixels = read_pgm(pgm_path)
    res = float(meta["resolution"])
    ox, oy = (float(v) for v in meta["origin"].strip("[]").split(",")[:2])
    thresh = float(meta.get("occupied_thresh", 0.65))
    negate = meta.get("negate", "0") == "1"

    # same rule as nav2 map_server: occupancy = (255 - p) / 255, flipped when negate is set
    def occupied(p: int) -> bool:
        return (p if negate else 255 - p) / 255 > thresh

    occ = [[occupied(pixels[r * width + c]) for c in range(width)] for r in range(height)]
    boxes = rectangles(occ)
    # image row 0 is the top of the map (largest y)
    walls = "".join(box_sdf(i, ox + (c0 + nc / 2) * res, oy + (height - r0 - nr / 2) * res, nc * res, nr * res)
                    for i, (r0, c0, nr, nc) in enumerate(boxes))
    name = os.path.splitext(os.path.basename(out_path))[0]
    with open(out_path, "w") as f:
        f.write(world_sdf(name, walls))
    return len(boxes)


def _self_check() -> None:
    occ = [[True, True, False], [True, True, False], [False, False, True]]
    assert sorted(rectangles(occ)) == [(0, 0, 2, 2), (2, 2, 1, 1)]


if __name__ == "__main__":
    _self_check()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("yaml", nargs="?", default="room.yaml")
    parser.add_argument("-o", "--out", help="output .sdf (default: <yaml name>.sdf); its name is the world name")
    args = parser.parse_args()
    out = args.out or os.path.splitext(args.yaml)[0] + ".sdf"
    n = convert(args.yaml, out)
    print(f"wrote {out} with {n} wall boxes")
