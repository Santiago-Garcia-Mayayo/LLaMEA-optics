"""Double-Gauss lens problem instances (24 mixed variables) on lensgopt (K. Antonov).

All instances share the simulator, the starting design and the 24 variables, and differ only in
the specification:

  original    - the group's current setting (LDG-EA / Yotam Lev thesis): F/3 (entrance pupil
                33.33 mm at EFL 99.5 mm), field +-14 deg, one min thickness 0.1 mm for every gap,
                edge >= 0.25 mm for every gap, FWD >= 3 mm. Thickness min enforced by box bounds.
  realistic   - DRAFT manufacturability spec, separate glass/air limits, EFFL with tolerance.
                Values to be confirmed with Kirill Antonov / Teus Tukker before use.
  fast        - original tolerances, faster lens: F/2.4 (entrance pupil 41.5 mm).
  wide        - original tolerances, wider field: +-20 deg.
  fast_wide   - both (F/2.4, +-20 deg).

Every evaluation returns the raw physical quantities, from which both arms are built:
  penalty(x)      -> the group's summed loss (identical to lensgopt for the original instance)
  constrained(x)  -> (f, g): f = RMS^2 spot (inf if any ray is lost), g = normalised violations
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ._threads import few_cores

with few_cores():          # JAX's CPU client sizes its thread pools now (see _threads.py)
    jax.devices()

import json
import os
import warnings


def _import_lensgopt():
    """lensgopt (K. Antonov's CameraLensSimulation) either installed, or found via
    $LENSGOPT_ROOT, or as a sibling folder ../CameraLensSimulation of this repo."""
    try:
        import lensgopt  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("LENSGOPT_ROOT", "")] + [
        os.path.join(here, "..", "..", name) for name in ("CameraLensSimulation", "CameraLensSimulation-main")
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "lensgopt")):
            sys.path.insert(0, os.path.abspath(c))
            return
    raise ImportError("lensgopt not found: pip install -e the CameraLensSimulation repo, "
                      "or set LENSGOPT_ROOT to its folder")


_import_lensgopt()
import lensgopt.optics.optics as optics  # noqa: E402
import lensgopt.optics.shapes as shapes  # noqa: E402
import lensgopt.optics.model as model  # noqa: E402
import lensgopt.optics.loss as lgloss  # noqa: E402

from .double_gauss_objective import DoubleGaussObjective  # noqa: E402

# gap i = distance from surface i to surface i+1 (distances_z), medium after surface i
GLASS_GAPS = (0, 2, 3, 6, 7, 9)
AIR_GAPS = (1, 4, 5, 8)        # 4, 5: around the aperture stop, fixed in the 24-var problem
SENSOR_GAP = 10  # last surface -> sensor: centre governed by FWD, edge by air_min_edge


@dataclass(frozen=True)
class LensSpec:
    name: str
    glass_min_ct: float          # min centre thickness of glass elements [mm]
    air_min_ct: float            # min centre thickness of air gaps [mm]
    glass_min_edge: float        # min edge thickness of glass elements [mm]
    air_min_edge: float          # min edge thickness of air gaps [mm]
    fwd_min: float               # min free working distance [mm]
    effl_target: float = 99.5    # [mm]
    effl_tol: float | None = None  # None -> squared equality (original); else |dEFFL| <= tol
    thickness_lb: float = 0.1    # box lower bound for optimised thicknesses [mm]
    entrance_diameter: float | None = None   # [mm]; None -> lens file (33.33 mm, F/3)
    max_field_deg: float | None = None       # [deg]; None -> lens file (14 deg)
    lensgopt_penalty: bool = False           # penalty arm: lensgopt's own summed loss
    source: dict = field(default_factory=dict)


ORIGINAL = LensSpec(
    name="original", glass_min_ct=0.1, air_min_ct=0.1, glass_min_edge=0.25, air_min_edge=0.25,
    fwd_min=3.0, effl_tol=0.5, thickness_lb=0.1, lensgopt_penalty=True,
    source={"all": "lensgopt create_double_gauss(); same numbers as CODE V example in "
                   "Juergens OPTI 517 (EFL 6.43 mm lens)",
            "effl_tol": "lensgopt penalises (EFFL-99.5)^2 with no tolerance; the penalty arm keeps "
                        "that exactly, the feasibility criterion (both arms) uses |dEFFL| <= 0.5 mm"},
)

REALISTIC_DRAFT = LensSpec(
    name="realistic_draft", glass_min_ct=2.0, air_min_ct=0.5, glass_min_edge=1.0, air_min_edge=0.1,
    fwd_min=3.0, effl_tol=0.5, thickness_lb=0.1,
    source={
        "glass_min_ct": "OSLO DBLGAUSS example: glass 1-15 mm at EFL 50 mm, scaled x2 to EFL ~100 mm",
        "air_min_ct": "OSLO DBLGAUSS: air >= 0.5 mm; reference Zemax design uses 0.5 mm gaps",
        "glass_min_edge": "Edmund Optics: edge >~0.7 mm at oversized diameter; rounded to 1.0 mm",
        "air_min_edge": "no edge contact between elements (small clearance); to confirm",
        "fwd_min": "unchanged from original",
        "effl_tol": "+-0.5 % of EFL, consistent with radius/index tolerances; to confirm",
    },
)

def _variant(name, **kw):
    """Original tolerances with a different optical specification."""
    from dataclasses import replace
    return replace(ORIGINAL, name=name, source={**ORIGINAL.source, "variant": kw}, **kw)


# F/2.4 at EFL 99.5 mm -> entrance pupil 41.5 mm; +-20 deg field. Classic Double-Gauss camera
# lenses reach about F/2 and +-23 deg, so both stay within what the design form can do.
FAST = _variant("fast", entrance_diameter=41.5)
WIDE = _variant("wide", max_field_deg=20.0)
FAST_WIDE = _variant("fast_wide", entrance_diameter=41.5, max_field_deg=20.0)

INSTANCES = {s.name: s for s in (ORIGINAL, REALISTIC_DRAFT, FAST, WIDE, FAST_WIDE)}


class LensProblem:
    def __init__(self, spec: LensSpec | str = "original"):
        self.spec = INSTANCES[spec] if isinstance(spec, str) else spec
        self.name = f"double_gauss/{self.spec.name}"
        self.obj = DoubleGaussObjective(enable_grad=False, enable_hessian=False)
        c = self.obj.cnst
        if self.spec.entrance_diameter is not None:
            c = c._replace(aperture=c.aperture._replace(max_d=float(self.spec.entrance_diameter)))
        if self.spec.max_field_deg is not None:
            c = c._replace(lens_field=c.lens_field._replace(max_field=float(self.spec.max_field_deg)))
        self.obj.cnst = c   # the objective's loss reads obj.cnst when it is traced
        self.cnst = c
        f = self.obj.factory
        self.nx = self.obj.n_x
        self.dim = self.obj.n_theta
        self.integer_indices = self.obj.integer_indices
        lb, ub = self.obj.bounds()
        lb[self.obj.n_c:self.nx] = self.spec.thickness_lb
        self.lb, self.ub = lb, ub
        self._dist_opt_idx = np.asarray(f.distances_optimized_idx)
        self._eval = jax.jit(self._raw_jax)
        self._eval_batch = jax.jit(jax.vmap(self._raw_jax))
        self.n_rays = self._count_rays()
        self.scales = self._load_scales()

    # ------------------------------------------------------------------ physics
    def _count_rays(self) -> int:
        """Number of rays traced for the RMS (pupil grid x fields x wavelengths + chief rays)."""
        c = self.cnst
        xc, xm = self.obj.init_from_templates()
        curv, dist, iors = self.obj.factory.unpack_full_with_iors(
            jnp.asarray(np.clip(xc, self.lb[: self.nx], self.ub[: self.nx])), jnp.asarray(xm))
        props = optics.compute_optical_properties(c, model.LensSystemVariables(curv, dist, iors))
        ang = jnp.array(c.field_factors) * c.lens_field.max_field
        rays = optics.RayRMSSpotSize.create_grid_rays_angle_field_for_rms_spot(
            n_rays_row=c.n_rays_row, n_wl=len(c.wavelengths), r_ep=props.entrance_pupil_diameter / 2,
            z_ep=props.entrance_pupil_z, angles_z_deg=ang)
        rays = optics.RayRMSSpotSize.append_ray(
            rays, optics.RayRMSSpotSize.create_chief_rays(len(c.wavelengths), z_ep=props.entrance_pupil_z,
                                                          angles_z_deg=ang))
        return int(rays.mask.sum())

    def _raw_jax(self, x, mats):
        f = self.obj.factory
        curv, dist, iors = f.unpack_full_with_iors(x, mats)
        comps = lgloss.loss(self.cnst, curv, dist, iors)  # [rms2, 10*drop, edge, axis, fwd, effl]
        vars_ = model.LensSystemVariables(curv, dist, iors)
        props = optics.compute_optical_properties(self.cnst, vars_)
        vz = model.LensSystemVariables.resolve_verticies_z(dist)
        sensor = shapes.Spheric(props.sensor_z, jnp.inf)
        edge = self._edge_per_gap(curv, vz, iors, props, sensor)
        centre = vz[1:] - vz[:-1]
        fwd = sensor.vertex_z - vz[-1]
        return comps, edge, centre, fwd, props.effective_focal_length

    def _edge_per_gap(self, curv, vz, iors_, props, sensor):
        """Min axial distance between consecutive surfaces along the marginal edge rays
        (same rays and tracing as lensgopt.edge_thickness_constraint_inline), per gap."""
        c = self.cnst
        rays = optics.RayRMSSpotSize.create_edge_thickness_rays(
            c.edge_thickness_field_factors, c.edge_thickness_entrance_pupil_factors,
            props.entrance_pupil_z, props.entrance_pupil_diameter, c.lens_field)
        iors = iors_[..., 0][:, None]
        out, off, z0 = [], 0, None
        for i in range(c.num_surfaces):
            cls = shapes.get_surface_class_by_name(c.surface_types[i])
            k = c.num_parameters_per_surface[i]
            surf = cls.create(vz[i], curv[off: off + k]); off += k
            vi, o1 = surf.ray_surface_intersection(rays, props.limiting_vertices_z[i])
            vr, d1 = optics.refract_rays(rays, o1, surf, iors[i], iors[i + 1], props.up_corners_y[i])
            m = rays.mask & vi & vr
            rays = optics.RayRMSSpotSize(o=o1, d=d1, mask=m)
            z1 = o1[..., 2].flatten()
            if z0 is not None:
                out.append(jnp.min(jnp.where(m.flatten(), z1 - z0, jnp.inf)))
            z0 = z1
        _, o_s = sensor.ray_surface_intersection(rays, sensor.vertex_z)
        out.append(jnp.min(jnp.where(rays.mask.flatten(), o_s[..., 2].flatten() - z0, jnp.inf)))
        return jnp.stack(out)  # (num_surfaces,) gaps: 0..9 between surfaces, 10 = last surface -> sensor

    # ------------------------------------------------------------------ interface
    def seal(self, theta):
        return self.obj.project_theta(theta, lb=self.lb, ub=self.ub)

    def evaluate(self, theta) -> dict:
        t = self.seal(theta)
        comps, edge, centre, fwd, effl = self._eval(jnp.asarray(t[: self.nx]), jnp.asarray(t[self.nx:]))
        comps = np.asarray(comps)
        return {
            "theta": t, "rms2": float(comps[0]), "n_dropped": float(comps[1] / 10.0),
            "lensgopt_terms": comps, "edge": np.asarray(edge), "centre": np.asarray(centre),
            "fwd": float(fwd), "effl": float(effl),
        }

    def raw_violations(self, e: dict) -> np.ndarray:
        """Violations in physical units (>= 0, 0 = satisfied): fraction of rays lost, then mm."""
        s = self.spec
        g_rays = [min(1.0, e["n_dropped"] / self.n_rays)]  # graded hidden constraint (f = inf if > 0)
        g_ct = [max(0.0, s.glass_min_ct - e["centre"][i]) for i in GLASS_GAPS]
        g_ct += [max(0.0, s.air_min_ct - e["centre"][i]) for i in AIR_GAPS]
        g_ed = [max(0.0, s.glass_min_edge - e["edge"][i]) for i in GLASS_GAPS]
        g_ed += [max(0.0, s.air_min_edge - e["edge"][i]) for i in AIR_GAPS + (SENSOR_GAP,)]
        g_fwd = [max(0.0, s.fwd_min - e["fwd"])]
        g_effl = [max(0.0, abs(e["effl"] - s.effl_target) - (s.effl_tol or 0.0))]
        return np.array(g_rays + g_ct + g_ed + g_fwd + g_effl, float)

    MAX_CHUNK = 256

    def evaluate_batch(self, Theta) -> list:
        """Vectorised evaluate(): one jax.vmap call per chunk. Chunks are padded to a power of
        two (repeating the last row) so that only log2(MAX_CHUNK)+1 batch shapes are ever
        compiled; padded rows are discarded."""
        T = np.array([self.seal(t) for t in np.atleast_2d(Theta)])
        out = []
        for start in range(0, len(T), self.MAX_CHUNK):
            chunk = T[start:start + self.MAX_CHUNK]
            n = len(chunk)
            size = 1 << (n - 1).bit_length()
            padded = np.vstack([chunk, np.repeat(chunk[-1:], size - n, axis=0)]) if size > n else chunk
            comps, edge, centre, fwd, effl = (np.asarray(a) for a in self._eval_batch(
                jnp.asarray(padded[:, : self.nx]), jnp.asarray(padded[:, self.nx:])))
            for j in range(n):
                out.append({
                    "theta": chunk[j], "rms2": float(comps[j, 0]), "n_dropped": float(comps[j, 1] / 10.0),
                    "lensgopt_terms": comps[j], "edge": edge[j], "centre": centre[j],
                    "fwd": float(fwd[j]), "effl": float(effl[j]),
                })
        return out

    def evaluate_all_batch(self, Theta):
        """Vectorised evaluate_all(): list of (f, v, penalty)."""
        res = []
        for e in self.evaluate_batch(Theta):
            f = np.inf if e["n_dropped"] > 0 else e["rms2"]
            res.append((f, self.violations(e), self.penalty_from(e)))
        return res

    def violations(self, e: dict) -> np.ndarray:
        """Normalised violations: raw / s_j, s_j = median positive raw violation of constraint j
        over 2000 seeded uniform samples (data/lens_violation_scales.json), same rule as CEC."""
        return self.raw_violations(e) / self.scales

    def _load_scales(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lens_violation_scales.json")
        if os.path.exists(path):
            with open(path) as fh:
                d = json.load(fh)
            if self.spec.name in d:
                sc = np.asarray(d[self.spec.name]["scales"], float)
                assert sc.size == len(self.CONSTRAINT_NAMES)
                return sc
        warnings.warn(f"no stored violation scales for lens/{self.spec.name}: run "
                      "problems/scripts/compute_lens_scales.py; using 1.0")
        return np.ones(len(self.CONSTRAINT_NAMES))

    CONSTRAINT_NAMES = (
        ["rays_lost"] + [f"ct_glass_{i}" for i in GLASS_GAPS] + [f"ct_air_{i}" for i in AIR_GAPS]
        + [f"edge_glass_{i}" for i in GLASS_GAPS] + [f"edge_air_{i}" for i in AIR_GAPS + (10,)]
        + ["fwd", "effl"]
    )

    def penalty_from(self, e: dict) -> float:
        """Summed-penalty loss from one evaluation. For `original` this is exactly lensgopt's loss."""
        if self.spec.lensgopt_penalty:
            return float(e["lensgopt_terms"].sum())
        s = self.spec
        sq = lambda a: float(np.sum(np.square(np.minimum(0.0, a))))  # noqa: E731
        c, ed = e["centre"], e["edge"]
        pen = 10 * e["n_dropped"]
        pen += sq(c[list(GLASS_GAPS)] - s.glass_min_ct) + sq(c[list(AIR_GAPS)] - s.air_min_ct)
        pen += sq(ed[list(GLASS_GAPS)] - s.glass_min_edge)
        pen += sq(ed[list(AIR_GAPS) + [SENSOR_GAP]] - s.air_min_edge)
        pen += sq(np.array([e["fwd"] - s.fwd_min]))
        pen += max(0.0, abs(e["effl"] - s.effl_target) - (s.effl_tol or 0.0)) ** 2
        return e["rms2"] + pen

    def evaluate_all(self, theta):
        """One simulation -> (f, v, penalty). f = inf if any ray is lost."""
        e = self.evaluate(theta)
        f = np.inf if e["n_dropped"] > 0 else e["rms2"]
        return f, self.violations(e), self.penalty_from(e)

    def penalty(self, theta) -> float:
        return self.penalty_from(self.evaluate(theta))

    def constrained(self, theta):
        e = self.evaluate(theta)
        f = np.inf if e["n_dropped"] > 0 else e["rms2"]
        return f, self.violations(e)

    def template(self) -> np.ndarray:
        xc, xm = self.obj.init_from_templates()
        return self.seal(np.r_[xc, xm])