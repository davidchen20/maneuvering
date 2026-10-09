"""
docking_planner.py

A Hybrid-A*-style planner for a diff-drive wheelchair docking into a
recessed bay (the "bus ramp" case: you can't approach face-first, you
have to turn around and back in), reaching an arbitrary target pose
(x, y, theta) subject to a minimum turning radius and wall obstacles.

This generalizes the corridor 3-point-turn sim: instead of "reverse
heading in a strip," the goal is "reach this exact pose over there,"
which pure in-place pivoting can never do (it can't translate you).

Motion model: same as before -- same steering lock (curvature) held per
primitive, gear (forward/reverse) determines translation direction only.
Search: A* over a discretized (x, y, theta) grid using 6 motion
primitives (straight / left / right, each forward and reverse), with a
switch-of-gear cost penalty so the search naturally prefers fewer
direction changes (i.e. fewer "points" in the turn).

Usage:
    python3 docking_planner.py --radius 0.5 --out dock1
    python3 docking_planner.py --radius 0.5 --animate --out dock1
"""

import argparse
import csv
import heapq
import math
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass
class Footprint:
    length: float = 1.1
    width: float = 0.7


def footprint_corners(x, y, theta, fp: Footprint):
    hl, hw = fp.length / 2.0, fp.width / 2.0
    local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    c, s = math.cos(theta), math.sin(theta)
    return [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]


@dataclass
class BayEnv:
    """Corridor of width W along x, with a rectangular bay recessed
    into the bottom wall between x=[bx1,bx2], depth `depth` below y=0."""
    W: float
    bx1: float
    bx2: float
    depth: float

    def point_free(self, x, y):
        if self.bx1 <= x <= self.bx2:
            return -self.depth <= y <= self.W
        return 0.0 <= y <= self.W

    def footprint_free(self, x, y, theta, fp: Footprint):
        return all(self.point_free(cx, cy) for cx, cy in footprint_corners(x, y, theta, fp))


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def propagate(x, y, theta, gear, kappa, step_len, n_sub=6):
    """Roll out one motion primitive, returning the list of intermediate
    (x, y, theta) states (for collision checking + trajectory logging)."""
    ds = step_len / n_sub
    states = []
    for _ in range(n_sub):
        x += gear * ds * math.cos(theta)
        y += gear * ds * math.sin(theta)
        theta += kappa * ds
        theta = wrap_angle(theta)
        states.append((x, y, theta))
    return states


def plan(start, goal, env: BayEnv, fp: Footprint, radius=0.5, step_len=0.2,
         xy_res=0.12, theta_bins=24, switch_penalty=0.4, max_expansions=300000):
    kappa_mag = 1.0 / radius
    primitives = [(1, 0.0), (1, kappa_mag), (1, -kappa_mag),
                  (-1, 0.0), (-1, kappa_mag), (-1, -kappa_mag)]

    def key(x, y, theta):
        tb = int(round(wrap_angle(theta) / (2 * math.pi) * theta_bins)) % theta_bins
        return (int(round(x / xy_res)), int(round(y / xy_res)), tb)

    def heuristic(x, y, theta, gx, gy, gtheta):
        d = math.hypot(gx - x, gy - y)
        ang = abs(wrap_angle(gtheta - theta))
        return d + 0.35 * ang

    gx, gy, gtheta = goal
    start_key = key(*start)
    goal_tol_xy = xy_res * 1.5
    goal_tol_th = math.radians(20)

    open_heap = [(heuristic(*start, gx, gy, gtheta), 0.0, start, None, None)]
    best_g = {start_key: 0.0}
    came_from = {}  # key -> (parent_key, edge_states, gear)
    node_state = {start_key: start}
    expansions = 0
    goal_key_found = None

    while open_heap and expansions < max_expansions:
        f, g, state, last_gear, _ = heapq.heappop(open_heap)
        x, y, theta = state
        k = key(x, y, theta)
        if g > best_g.get(k, math.inf) + 1e-9:
            continue
        expansions += 1

        if (math.hypot(gx - x, gy - y) < goal_tol_xy and
                abs(wrap_angle(gtheta - theta)) < goal_tol_th):
            goal_key_found = k
            node_state[k] = state
            break

        for gear, kappa in primitives:
            edge = propagate(x, y, theta, gear, kappa, step_len)
            if not all(env.footprint_free(sx, sy, sth, fp) for sx, sy, sth in edge):
                continue
            nx, ny, ntheta = edge[-1]
            nk = key(nx, ny, ntheta)
            switch = switch_penalty if (last_gear not in (None, gear)) else 0.0
            ng = g + step_len + switch
            if ng < best_g.get(nk, math.inf) - 1e-9:
                best_g[nk] = ng
                came_from[nk] = (k, edge, gear)
                node_state[nk] = (nx, ny, ntheta)
                nf = ng + heuristic(nx, ny, ntheta, gx, gy, gtheta)
                heapq.heappush(open_heap, (nf, ng, (nx, ny, ntheta), gear, nk))

    if goal_key_found is None:
        return None, expansions

    # reconstruct
    path_states = [start]
    gear_switches = 0
    prev_gear = None
    k = goal_key_found
    edges = []
    while k in came_from:
        pk, edge, gear = came_from[k]
        edges.append((edge, gear))
        k = pk
    edges.reverse()
    for edge, gear in edges:
        if prev_gear is not None and gear != prev_gear:
            gear_switches += 1
        prev_gear = gear
        path_states.extend(edge)

    return {"states": path_states, "gear_switches": gear_switches}, expansions


def write_csv(states, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "theta_deg"])
        for x, y, th in states:
            w.writerow([f"{x:.4f}", f"{y:.4f}", f"{math.degrees(th):.2f}"])


def draw_env(ax, env: BayEnv, x_max):
    ax.axhline(env.W, color="#3a4452", linewidth=4)
    ax.plot([-0.6, env.bx1], [0, 0], color="#3a4452", linewidth=4)
    ax.plot([env.bx2, x_max], [0, 0], color="#3a4452", linewidth=4)
    ax.plot([env.bx1, env.bx1], [0, -env.depth], color="#3a4452", linewidth=4)
    ax.plot([env.bx2, env.bx2], [0, -env.depth], color="#3a4452", linewidth=4)
    ax.plot([env.bx1, env.bx2], [-env.depth, -env.depth], color="#3a4452", linewidth=4)


def plot_static(result, env, fp, goal, path):
    states = result["states"]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0e1116")
    ax.set_facecolor("#0a0d12")
    x_max = max(s[0] for s in states) + 0.6
    draw_env(ax, env, x_max)

    xs = [s[0] for s in states]
    ys = [s[1] for s in states]
    ax.plot(xs, ys, color="#f2a541", linewidth=1.2, alpha=0.8)

    for i in range(0, len(states), max(1, len(states) // 18)):
        x, y, th = states[i]
        corners = footprint_corners(x, y, th, fp)
        ax.add_patch(Polygon(corners, closed=True, facecolor="#3ecf8e",
                              edgecolor="#3ecf8e", alpha=0.18))
    last = states[-1]
    ax.add_patch(Polygon(footprint_corners(*last, fp), closed=True,
                          facecolor="none", edgecolor="white", linewidth=1.8))

    gx, gy, gth = goal
    ax.plot(gx, gy, marker="x", color="#ef5757", markersize=10, mew=2)

    ax.set_aspect("equal")
    ax.set_xlim(-0.8, x_max)
    ax.set_ylim(-env.depth - 0.4, env.W + 0.4)
    ax.set_title(f"Docking maneuver  |  R={ARGS_RADIUS:.2f} m  |  "
                 f"{result['gear_switches']} gear switch(es)",
                 color="#e7ecf2", fontsize=11, family="monospace")
    ax.tick_params(colors="#8b96a5")
    for spine in ax.spines.values():
        spine.set_color("#2a323d")

    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def animate(result, env, fp, goal, path):
    states = result["states"]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0e1116")
    ax.set_facecolor("#0a0d12")
    x_max = max(s[0] for s in states) + 0.6
    draw_env(ax, env, x_max)
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, x_max)
    ax.set_ylim(-env.depth - 0.4, env.W + 0.4)
    ax.tick_params(colors="#8b96a5")
    for spine in ax.spines.values():
        spine.set_color("#2a323d")

    trail, = ax.plot([], [], color="#f2a541", linewidth=1.2, alpha=0.8)
    chair = Polygon([(0, 0)] * 4, closed=True, facecolor="#3ecf8e", edgecolor="#3ecf8e", alpha=0.6)
    ax.add_patch(chair)

    step = max(1, len(states) // 220)
    frames = list(range(0, len(states), step))

    def update(i):
        idx = frames[i]
        x, y, th = states[idx]
        trail.set_data([s[0] for s in states[:idx + 1]], [s[1] for s in states[:idx + 1]])
        chair.set_xy(footprint_corners(x, y, th, fp))
        return trail, chair

    anim = FuncAnimation(fig, update, frames=len(frames), interval=25)
    anim.save(path, writer=PillowWriter(fps=30))
    plt.close(fig)


ARGS_RADIUS = 0.5


def main():
    global ARGS_RADIUS
    parser = argparse.ArgumentParser(description="Docking planner: bus-ramp-style bay")
    parser.add_argument("--radius", type=float, default=0.5, help="min turn radius (m)")
    parser.add_argument("--switch-penalty", type=float, default=0.4, help="cost added per gear reversal")
    parser.add_argument("--goal-heading-deg", type=float, default=180.0,
                         help="desired final heading in degrees (0=facing +x, 180=facing -x, "
                              "90=facing +y, -90/270=facing -y)")
    parser.add_argument("--corridor-width", type=float, default=1.4)
    parser.add_argument("--bay-x1", type=float, default=2.0)
    parser.add_argument("--bay-x2", type=float, default=3.2)
    parser.add_argument("--bay-depth", type=float, default=1.3)
    parser.add_argument("--out", default="dock")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    ARGS_RADIUS = args.radius

    fp = Footprint()
    env = BayEnv(W=args.corridor_width, bx1=args.bay_x1, bx2=args.bay_x2, depth=args.bay_depth)

    start = (0.0, env.W / 2.0, 0.0)  # in the corridor, facing toward the bay
    bay_cx = (env.bx1 + env.bx2) / 2.0
    goal = (bay_cx, -env.depth / 2.0, math.radians(args.goal_heading_deg))  # backed in, facing back out

    result, expansions = plan(start, goal, env, fp, radius=args.radius, switch_penalty=args.switch_penalty)

    if result is None:
        print(f"NO PATH FOUND after {expansions} expansions.")
        print("Try a smaller --radius, a wider bay, or a deeper bay.")
        return

    write_csv(result["states"], f"{args.out}_trajectory.csv")
    plot_static(result, env, fp, goal, f"{args.out}_viz.png")
    if args.animate:
        animate(result, env, fp, goal, f"{args.out}_viz.gif")

    print(f"expansions      : {expansions}")
    print(f"path points     : {len(result['states'])}")
    print(f"gear switches   : {result['gear_switches']}  (~ an {result['gear_switches']+1}-point turn)")
    print(f"final pose      : x={result['states'][-1][0]:.2f} y={result['states'][-1][1]:.2f} "
          f"theta={math.degrees(result['states'][-1][2]):.1f} deg")
    print(f"wrote           : {args.out}_trajectory.csv, {args.out}_viz.png"
          + (f", {args.out}_viz.gif" if args.animate else ""))


if __name__ == "__main__":
    main()