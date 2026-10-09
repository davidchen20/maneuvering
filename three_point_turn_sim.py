"""
three_point_turn_sim.py

Standalone prototype (no ROS dependency) of a diff-drive wheelchair
reversing heading inside a width-constrained corridor. Compares an
in-place PIVOT against a THREE-POINT TURN under the same space constraint.

Model: unicycle kinematics (x, y, theta), rectangular footprint.
Three-point turn = same steering lock (omega) held throughout; only the
drive direction alternates (forward / reverse / forward), each phase
sweeping 60 degrees of heading so the three phases sum to 180.

Usage:
    python3 three_point_turn_sim.py --mode 3pt   --width 1.4 --radius 0.55 --out run1
    python3 three_point_turn_sim.py --mode pivot --width 1.4 --out run1_pivot
    python3 three_point_turn_sim.py --mode 3pt --width 1.4 --radius 0.55 --animate --out run1

Outputs:
    <out>_trajectory.csv   -- t, x, y, theta_deg, phase, hit
    <out>_viz.png          -- static plot: corridor walls, trail, footprint snapshots
    <out>_viz.gif          -- animation (only if --animate is passed)
"""

import argparse
import csv
import math
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class Footprint:
    length: float = 1.1  # m
    width: float = 1.1   # m


@dataclass
class State:
    t: float
    x: float
    y: float
    theta: float
    phase: int
    hit: bool


@dataclass
class SimResult:
    traj: list = field(default_factory=list)
    collided: bool = False
    min_clearance: float = 1e9
    corridor_width: float = 0.0


def footprint_corners(x, y, theta, fp: Footprint):
    hl, hw = fp.length / 2.0, fp.width / 2.0
    local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    c, s = math.cos(theta), math.sin(theta)
    return [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]


def record_state(res: SimResult, t, x, y, theta, phase, W, fp: Footprint):
    corners = footprint_corners(x, y, theta, fp)
    hit = any(cy < 0.0 or cy > W for _, cy in corners)
    local_clear = min([W / 2.0] + [min(cy, W - cy) for _, cy in corners])
    if hit:
        res.collided = True
    res.min_clearance = min(res.min_clearance, local_clear)
    res.traj.append(State(t, x, y, theta, phase, hit))


def simulate_pivot(W, fp: Footprint) -> SimResult:
    res = SimResult(corridor_width=W)
    x, y, theta, t = 0.0, W / 2.0, 0.0, 0.0
    dt = 0.02
    omega = math.pi  # rad/s, in-place rotation
    record_state(res, t, x, y, theta, 0, W, fp)
    turned = 0.0
    while turned < math.pi:
        theta += omega * dt
        turned += omega * dt
        t += dt
        record_state(res, t, x, y, theta, 0, W, fp)
    return res


def simulate_single_arc(W, R, fp: Footprint) -> SimResult:
    """Continuous single-arc U-turn: constant forward v, constant omega,
    no reversal. This is the fair baseline for the 3-point turn -- both use
    the same turn radius R, but this one never backs up."""
    res = SimResult(corridor_width=W)
    v0 = 0.35
    dt = 0.02
    omega0 = v0 / R
    x, y, theta, t = 0.0, W / 2.0, 0.0, 0.0
    record_state(res, t, x, y, theta, 1, W, fp)
    turned = 0.0
    while turned < math.pi:
        x += v0 * math.cos(theta) * dt
        y += v0 * math.sin(theta) * dt
        theta += omega0 * dt
        turned += omega0 * dt
        t += dt
        record_state(res, t, x, y, theta, 1, W, fp)
    return res


def simulate_three_point_turn(W, R, fp: Footprint) -> SimResult:
    res = SimResult(corridor_width=W)
    v0 = 0.35        # m/s
    dt = 0.02         # s
    omega0 = v0 / R
    phase_angle = math.pi / 3.0  # 60 deg per phase, x3 = 180
    v_signs = (1, -1, 1)

    x, y, theta, t = 0.0, W / 2.0, 0.0, 0.0
    record_state(res, t, x, y, theta, 1, W, fp)

    for p, sign in enumerate(v_signs, start=1):
        v = sign * v0
        turned = 0.0
        while turned < phase_angle:
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += omega0 * dt
            turned += omega0 * dt
            t += dt
            record_state(res, t, x, y, theta, p, W, fp)
    return res


def write_csv(res: SimResult, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "theta_deg", "phase", "hit"])
        for s in res.traj:
            w.writerow([f"{s.t:.3f}", f"{s.x:.4f}", f"{s.y:.4f}",
                        f"{math.degrees(s.theta):.2f}", s.phase, int(s.hit)])


def plot_static(res: SimResult, fp: Footprint, path: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#0e1116")
    ax.set_facecolor("#0a0d12")
    W = res.corridor_width

    max_x = max(s.x for s in res.traj)
    ax.set_xlim(-0.6, max_x + 0.6)
    ax.set_ylim(-0.15, W + 0.15)
    ax.set_aspect("equal")

    # walls
    ax.axhline(0, color="#3a4452", linewidth=4)
    ax.axhline(W, color="#3a4452", linewidth=4)

    # trail
    xs = [s.x for s in res.traj]
    ys = [s.y for s in res.traj]
    ax.plot(xs, ys, color="#f2a541", linewidth=1.2, alpha=0.7)

    # footprint snapshots
    for i in range(0, len(res.traj), 15):
        s = res.traj[i]
        corners = footprint_corners(s.x, s.y, s.theta, fp)
        color = "#ef5757" if s.hit else "#3ecf8e"
        poly = Polygon(corners, closed=True, facecolor=color, edgecolor=color, alpha=0.22)
        ax.add_patch(poly)

    # final pose
    last = res.traj[-1]
    corners = footprint_corners(last.x, last.y, last.theta, fp)
    ax.add_patch(Polygon(corners, closed=True, facecolor="none", edgecolor="white", linewidth=1.8))

    ax.set_title(title, color="#e7ecf2", fontsize=11, family="monospace")
    ax.tick_params(colors="#8b96a5")
    for spine in ax.spines.values():
        spine.set_color("#2a323d")

    status = "COLLISION" if res.collided else f"clear, min gap {res.min_clearance*100:.0f} cm"
    ax.text(0.01, 0.97, status, transform=ax.transAxes, color=("#ef5757" if res.collided else "#3ecf8e"),
            fontsize=10, family="monospace", va="top")

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def animate(res: SimResult, fp: Footprint, path: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#0e1116")
    ax.set_facecolor("#0a0d12")
    W = res.corridor_width
    max_x = max(s.x for s in res.traj)
    ax.set_xlim(-0.6, max_x + 0.6)
    ax.set_ylim(-0.15, W + 0.15)
    ax.set_aspect("equal")
    ax.axhline(0, color="#3a4452", linewidth=4)
    ax.axhline(W, color="#3a4452", linewidth=4)
    ax.set_title(title, color="#e7ecf2", fontsize=11, family="monospace")
    ax.tick_params(colors="#8b96a5")
    for spine in ax.spines.values():
        spine.set_color("#2a323d")

    trail_line, = ax.plot([], [], color="#f2a541", linewidth=1.2, alpha=0.7)
    chair_poly = Polygon([(0, 0)] * 4, closed=True, facecolor="#3ecf8e", edgecolor="#3ecf8e", alpha=0.5)
    ax.add_patch(chair_poly)

    step = max(1, len(res.traj) // 200)  # cap frames for a reasonable gif size
    frames = list(range(0, len(res.traj), step))

    def update(i):
        idx = frames[i]
        s = res.traj[idx]
        xs = [p.x for p in res.traj[: idx + 1]]
        ys = [p.y for p in res.traj[: idx + 1]]
        trail_line.set_data(xs, ys)
        corners = footprint_corners(s.x, s.y, s.theta, fp)
        chair_poly.set_xy(corners)
        color = "#ef5757" if s.hit else "#3ecf8e"
        chair_poly.set_facecolor(color)
        chair_poly.set_edgecolor(color)
        return trail_line, chair_poly

    anim = FuncAnimation(fig, update, frames=len(frames), interval=30, blit=False)
    anim.save(path, writer=PillowWriter(fps=30))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Three-point-turn maneuvering prototype")
    parser.add_argument("--mode", choices=["pivot", "3pt", "arc"], default="3pt")
    parser.add_argument("--width", type=float, default=1.4, help="corridor width (m)")
    parser.add_argument("--radius", type=float, default=0.55, help="turn radius, 3pt mode only (m)")
    parser.add_argument("--out", default="run")
    parser.add_argument("--animate", action="store_true", help="also write an animated GIF")
    args = parser.parse_args()

    fp = Footprint()
    if args.mode == "pivot":
        res = simulate_pivot(args.width, fp)
        title = f"PIVOT (in place)  |  corridor {args.width:.2f} m"
    elif args.mode == "arc":
        res = simulate_single_arc(args.width, args.radius, fp)
        title = f"SINGLE ARC U-TURN  |  corridor {args.width:.2f} m, R {args.radius:.2f} m"
    else:
        res = simulate_three_point_turn(args.width, args.radius, fp)
        title = f"3-POINT TURN  |  corridor {args.width:.2f} m, R {args.radius:.2f} m"

    write_csv(res, f"{args.out}_trajectory.csv")
    plot_static(res, fp, f"{args.out}_viz.png", title)
    if args.animate:
        animate(res, fp, f"{args.out}_viz.gif", title)

    print(f"mode           : {args.mode}")
    print(f"corridor width : {args.width} m")
    if args.mode in ("3pt", "arc"):
        print(f"turn radius    : {args.radius} m")
    print(f"footprint      : {fp.length} x {fp.width} m")
    print(f"steps          : {len(res.traj)}")
    print(f"collided       : {'YES' if res.collided else 'no'}")
    print(f"min clearance  : {res.min_clearance*100:.1f} cm")
    print(f"wrote          : {args.out}_trajectory.csv, {args.out}_viz.png"
          + (f", {args.out}_viz.gif" if args.animate else ""))


if __name__ == "__main__":
    main()
