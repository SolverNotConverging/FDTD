"""Subcell PEC geometry on the two dimensional Yee grid.

Fractions are geometric only.  Constitutive parameters continue to use the
solvers' existing material averaging.  Shape specifications contain plain
numbers so that simulation checkpoints remain pickleable.
"""

import numpy as np


def _geometry(shape):
    return (shape[1], shape[2]) if len(shape) == 3 else shape


def _inside(shape, x, y):
    kind, data = shape
    if kind == "rectangle":
        x0, x1, y0, y1 = data
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if kind == "circle":
        cx, cy, radius = data
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    if kind == "triangle":
        (ax, ay), (bx, by), (cx, cy) = data
        ab = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        bc = (cx - bx) * (y - by) - (cy - by) * (x - bx)
        ca = (ax - cx) * (y - cy) - (ay - cy) * (x - cx)
        scale = max((bx - ax) ** 2 + (by - ay) ** 2,
                    (cx - bx) ** 2 + (cy - by) ** 2,
                    (ax - cx) ** 2 + (ay - cy) ** 2)
        tolerance = 1e-12 * scale
        return ((ab >= -tolerance) & (bc >= -tolerance) & (ca >= -tolerance)
                | (ab <= tolerance) & (bc <= tolerance) & (ca <= tolerance))
    raise ValueError(f"Unknown PEC shape {kind!r}")


def _occupied(shapes, x, y):
    inside = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=bool)
    for shape in shapes:
        if len(shape) == 3 and shape[0] in {"PEC", "PMC"}:
            action, kind, data = shape
            hit = _inside((kind, data), x, y)
            inside = np.where(hit, action == "PEC", inside)
        else:  # checkpoints written before ordered conductor shapes
            inside |= _inside(shape, x, y)
    return inside


def _bounds(shape):
    shape = _geometry(shape)
    kind, data = shape
    if kind == "rectangle":
        x0, x1, y0, y1 = data
        return x0, x1, y0, y1
    if kind == "circle":
        cx, cy, radius = data
        return cx - radius, cx + radius, cy - radius, cy + radius
    points = np.asarray(data)
    return (points[:, 0].min(), points[:, 0].max(),
            points[:, 1].min(), points[:, 1].max())


def rasterize_pec(shapes, nx, ny, dx, dy, samples=48):
    """Return fluid area, x/y edge fractions, and fluid node flags.

    Interior quadrature avoids declaring a grazing contact to be a closed
    edge.  The fractions converge as ``samples`` increases; they are not
    rounded to a material-cell mask.
    """
    area = np.ones((nx, ny), dtype=float)
    x_edge = np.ones((nx, ny + 1), dtype=float)
    y_edge = np.ones((nx + 1, ny), dtype=float)
    nodes = np.ones((nx + 1, ny + 1), dtype=bool)
    if not shapes or not any(len(shape) == 2 or shape[0] == "PEC" for shape in shapes):
        return area, x_edge, y_edge, nodes
    bounds = [_bounds(shape) for shape in shapes
              if len(shape) == 2 or shape[0] == "PEC"]
    has_erasure = any(len(shape) == 3 and shape[0] == "PMC" for shape in shapes)
    cells = set()
    x_edges = set()
    y_edges = set()
    node_indices = set()
    for x0, x1, y0, y1 in bounds:
        ilo = max(0, int(np.floor(x0 / dx)))
        ihi = min(nx, int(np.ceil(x1 / dx)))
        jlo = max(0, int(np.floor(y0 / dy)))
        jhi = min(ny, int(np.ceil(y1 / dy)))
        cells.update((i, j) for i in range(ilo, ihi) for j in range(jlo, jhi))
        x_edges.update((i, j) for i in range(ilo, ihi) for j in range(jlo, jhi + 1))
        y_edges.update((i, j) for i in range(ilo, ihi + 1) for j in range(jlo, jhi))
        node_indices.update((i, j) for i in range(ilo, ihi + 1) for j in range(jlo, jhi + 1))
    offsets = (np.arange(samples) + 0.5) / samples
    for i, j in cells:
        if not has_erasure and any(all(bool(_inside(_geometry(shape), x, y)) for x, y in
                   ((i * dx, j * dy), ((i + 1) * dx, j * dy),
                    (i * dx, (j + 1) * dy), ((i + 1) * dx, (j + 1) * dy)))
               for shape in shapes):
            area[i, j] = 0.0
            continue
        xs = (i + offsets) * dx
        ys = (j + offsets) * dy
        area[i, j] = 1.0 - np.mean(_occupied(shapes, xs[:, None], ys[None, :]))
    for i, j in x_edges:
        if not has_erasure and any(bool(_inside(_geometry(shape), i * dx, j * dy))
                                   and bool(_inside(_geometry(shape), (i + 1) * dx,
                                                    j * dy)) for shape in shapes):
            x_edge[i, j] = 0.0
            continue
        xs = (i + offsets) * dx
        x_edge[i, j] = 1.0 - np.mean(_occupied(shapes, xs, j * dy))
    for i, j in y_edges:
        if not has_erasure and any(bool(_inside(_geometry(shape), i * dx, j * dy))
                                   and bool(_inside(_geometry(shape), i * dx,
                                                    (j + 1) * dy)) for shape in shapes):
            y_edge[i, j] = 0.0
            continue
        ys = (j + offsets) * dy
        y_edge[i, j] = 1.0 - np.mean(_occupied(shapes, i * dx, ys))
    for i, j in node_indices:
        nodes[i, j] = not bool(_occupied(shapes, i * dx, j * dy))
    unresolved = ((area > 0) & (area < 1)
                  & (x_edge[:, :-1] == 1) & (x_edge[:, 1:] == 1)
                  & (y_edge[:-1, :] == 1) & (y_edge[1:, :] == 1))
    if np.any(unresolved):
        raise ValueError("PEC feature is enclosed in a cell without crossing a Yee edge; refine the grid")
    zero_area_with_open_edge = ((area == 0)
                                & ((x_edge[:, :-1] > 0) | (x_edge[:, 1:] > 0)
                                   | (y_edge[:-1, :] > 0) | (y_edge[1:, :] > 0)))
    if np.any(zero_area_with_open_edge):
        raise ValueError("PEC cut area is below geometric sampling resolution; increase subpixel")
    return area, x_edge, y_edge, nodes


def enlarged_groups(area, x_edge, y_edge, threshold=0.5):
    """Merge small fluid cells with the best connected neighbor.

    Each returned group uses one magnetic unknown and the sum of its member
    areas.  Thus no artificial area is added to a cut cell.
    """
    nx, ny = area.shape
    parent = np.arange(nx * ny)

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, j in zip(*np.where((area > 0) & (area < threshold))):
        choices = []
        if i > 0 and y_edge[i, j] > 0 and area[i - 1, j] > 0:
            choices.append((area[i - 1, j], (i - 1) * ny + j))
        if i + 1 < nx and y_edge[i + 1, j] > 0 and area[i + 1, j] > 0:
            choices.append((area[i + 1, j], (i + 1) * ny + j))
        if j > 0 and x_edge[i, j] > 0 and area[i, j - 1] > 0:
            choices.append((area[i, j - 1], i * ny + j - 1))
        if j + 1 < ny and x_edge[i, j + 1] > 0 and area[i, j + 1] > 0:
            choices.append((area[i, j + 1], i * ny + j + 1))
        if choices:
            parent[i * ny + j] = root(max(choices)[1])
    while True:
        totals = {}
        members_by_root = {}
        for i, j in zip(*np.where(area > 0)):
            key = root(i * ny + j)
            totals[key] = totals.get(key, 0.0) + area[i, j]
            members_by_root.setdefault(key, []).append((i, j))
        small = [key for key, total in totals.items() if total < threshold]
        if not small:
            break
        changed = False
        for key in small:
            if root(key) != key:
                continue
            choices = []
            for i, j in members_by_root[key]:
                for ni, nj, opening in (
                        (i - 1, j, y_edge[i, j]),
                        (i + 1, j, y_edge[i + 1, j]),
                        (i, j - 1, x_edge[i, j]),
                        (i, j + 1, x_edge[i, j + 1])):
                    if (0 <= ni < nx and 0 <= nj < ny and opening > 0
                            and area[ni, nj] > 0):
                        other = root(ni * ny + nj)
                        if other != key:
                            choices.append((totals[other], other))
            if choices:
                parent[key] = max(choices)[1]
                changed = True
        if not changed:
            raise ValueError("A PEC cut cell has no fluid neighbor for enlargement")
    groups = {}
    for i, j in zip(*np.where(area > 0)):
        groups.setdefault(root(i * ny + j), []).append((i, j))
    return [members for members in groups.values() if len(members) > 1]


def enlarged_segments(open_fraction, axis, threshold=0.5):
    """Join short cut segments to a fluid neighbor along their axis."""
    shape = open_fraction.shape
    parent = np.arange(open_fraction.size)

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for index in np.ndindex(shape):
        fraction = open_fraction[index]
        if not 0 < fraction < threshold:
            continue
        choices = []
        for offset in (-1, 1):
            neighbor = list(index)
            neighbor[axis] += offset
            if 0 <= neighbor[axis] < shape[axis]:
                neighbor = tuple(neighbor)
                if open_fraction[neighbor] > 0:
                    choices.append((open_fraction[neighbor],
                                    np.ravel_multi_index(neighbor, shape)))
        if choices:
            parent[np.ravel_multi_index(index, shape)] = root(max(choices)[1])
    while True:
        totals = {}
        members_by_root = {}
        for index in np.ndindex(shape):
            if open_fraction[index] > 0:
                key = root(np.ravel_multi_index(index, shape))
                totals[key] = totals.get(key, 0.0) + open_fraction[index]
                members_by_root.setdefault(key, []).append(index)
        small = [key for key, total in totals.items() if total < threshold]
        if not small:
            break
        changed = False
        for key in small:
            if root(key) != key:
                continue
            choices = []
            for index in members_by_root[key]:
                for offset in (-1, 1):
                    neighbor = list(index)
                    neighbor[axis] += offset
                    if 0 <= neighbor[axis] < shape[axis]:
                        neighbor = tuple(neighbor)
                        if open_fraction[neighbor] > 0:
                            other = root(np.ravel_multi_index(neighbor, shape))
                            if other != key:
                                choices.append((totals[other], other))
            if choices:
                parent[key] = max(choices)[1]
                changed = True
        if not changed:
            raise ValueError("A PEC cut segment has no fluid neighbor for enlargement")
    groups = {}
    for index in np.ndindex(shape):
        if open_fraction[index] > 0:
            groups.setdefault(root(np.ravel_multi_index(index, shape)), []).append(index)
    return [members for members in groups.values() if len(members) > 1]


def pack_enlarged_groups(groups, fraction, material):
    """Contiguous CSR group indices and normalized mass weights for native loops."""
    offsets = [0]
    ii = []
    jj = []
    weights = []
    for members in groups:
        indices = np.asarray(members, dtype=np.int64)
        x, y = indices.T
        mass = fraction[x, y] * material[x, y]
        ii.extend(x)
        jj.extend(y)
        weights.extend(mass / mass.sum())
        offsets.append(len(ii))
    return (np.asarray(offsets, dtype=np.int64),
            np.asarray(ii, dtype=np.int64),
            np.asarray(jj, dtype=np.int64),
            np.asarray(weights, dtype=np.float64))
