from __future__ import annotations

from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import lensgopt.optics.loss as loss
import lensgopt.optics.model as model
import lensgopt.optics.optics as optics
import lensgopt.parsers.lens_creation as lens_creation
import lensgopt.visualization.lens_vis as lens_vis


class DoubleGaussObjective:
    """Objective wrapper for mixed-integer optimization on Double-Gauss."""

    def __init__(
        self,
        *,
        enable_grad: bool = True,
        enable_hessian: bool = True,
    ):
        self.cnst, self.vars_template, self.material_ids_template = (
            lens_creation.create_double_gauss(sensor_z_mode="paraxial_solve")
        )

        self.factory = loss.LossFactoryWithMaterials(
            loss_full=lambda flat_params_, dists_, iors_: loss.loss(
                self.cnst, flat_params_, dists_, iors_
            ),
            curvatures_template=self.vars_template.curvature_parameters,
            distances_template=self.vars_template.distances_z,
            material_ids_template=self.material_ids_template,
            curvatures_optimized_idx=jnp.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]),
            distances_optimized_idx=jnp.array([0, 1, 2, 3, 6, 7, 8, 9]),
            materials_optimized_idx=jnp.array([1, 3, 4, 7, 8, 10]),
            catalogs=self.cnst.ior_catalogs,
        )

        self._f_loss = self.factory.make_jit_loss()
        self._f_grad = self.factory.make_grad() if enable_grad else None
        self._f_hess = self.factory.make_hessian() if enable_hessian else None

    @property
    def n_c(self) -> int:
        return int(self.factory.sz_packed_curvatures)

    @property
    def n_d(self) -> int:
        return int(self.factory.sz_packed_distances)

    @property
    def n_m(self) -> int:
        return int(self.factory.sz_packed_materials)

    @property
    def n_x(self) -> int:
        return self.n_c + self.n_d

    @property
    def n_theta(self) -> int:
        return self.n_x + self.n_m

    @property
    def integer_indices(self) -> list[int]:
        return list(range(self.n_x, self.n_theta))

    def init_from_templates(self) -> tuple[np.ndarray, np.ndarray]:
        x_cont, x_mat = self.factory.init_from_templates()
        return np.asarray(x_cont, dtype=float), np.asarray(x_mat, dtype=float)

    def pack_theta(
        self, continuous_params: Sequence[float], glasses_ids: Sequence[float]
    ) -> np.ndarray:
        x = np.asarray(continuous_params, dtype=float).ravel()
        g = np.asarray(glasses_ids, dtype=float).ravel()
        return np.concatenate([x, g])

    def split_theta(self, theta: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float).ravel()
        x_cont = theta[: self.n_x]
        x_mat = np.rint(theta[self.n_x :]).astype(int)
        return x_cont, x_mat

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb_c = np.full(self.n_c, -1000.0, dtype=float)
        if self.n_c > 0:
            lb_c[0] = 0.0
        ub_c = np.full(self.n_c, 1000.0, dtype=float)

        internal_idx = set(int(i) for i in [0, 2, 3, 6, 7, 9])
        dist_opt_idx = np.asarray(self.factory.distances_optimized_idx, dtype=int)
        lb_d = np.full(self.n_d, float(self.cnst.target_axis_thickness), dtype=float)
        ub_d = np.array(
            [20.0 if int(i) in internal_idx else 50.0 for i in dist_opt_idx],
            dtype=float,
        )

        mat_opt_idx = np.asarray(self.factory.materials_optimized_idx, dtype=int)
        lb_g = np.zeros(self.n_m, dtype=float)
        ub_g = np.array(
            [float(len(self.cnst.ior_catalogs[int(i)]) - 1) for i in mat_opt_idx],
            dtype=float,
        )

        lb = np.concatenate([lb_c, lb_d, lb_g])
        ub = np.concatenate([ub_c, ub_d, ub_g])
        assert lb.shape == (self.n_theta,) and ub.shape == (self.n_theta,)
        return lb, ub

    def project_theta(
        self,
        theta: Sequence[float],
        *,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float).ravel()
        if lb is None or ub is None:
            lb, ub = self.bounds()
        theta = np.clip(theta, lb, ub)
        theta[self.integer_indices] = np.rint(theta[self.integer_indices])
        return theta

    def objective_cont_int(
        self, continuous_params: jnp.array, glasses_ids: jnp.array
    ) -> float:
        return float(self._f_loss(continuous_params, glasses_ids))

    def objective_theta(self, theta: Sequence[float]) -> float:
        return float(
            self._f_loss(jnp.asarray(theta[: self.n_x]), jnp.asarray(theta[self.n_x :]))
        )

    def gradient_cont_int(
        self, continuous_params: Sequence[float], glasses_ids: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._f_grad is None:
            raise RuntimeError("Gradient is disabled. Recreate with enable_grad=True.")
        g_cont = self._f_grad(jnp.asarray(continuous_params), jnp.asarray(glasses_ids))
        return np.asarray(g_cont)

    def hessian_cont_int(
        self, continuous_params: Sequence[float], glasses_ids: Sequence[float]
    ) -> np.ndarray:
        if self._f_hess is None:
            raise RuntimeError(
                "Hessian is disabled. Recreate with enable_hessian=True."
            )
        h = self._f_hess(jnp.asarray(continuous_params), jnp.asarray(glasses_ids))
        return np.asarray(h)

    def unpack_theta_to_design(
        self, theta: Sequence[float]
    ) -> tuple[model.LensSystemVariables, np.ndarray]:
        theta = self.project_theta(theta)
        x_cont, x_mat = self.split_theta(theta)
        curv, dist, iors = self.factory.unpack_full_with_iors(
            jnp.asarray(x_cont), jnp.asarray(x_mat)
        )
        _, _, material_ids_full = self.factory.unpack_full_with_materials(
            jnp.asarray(x_cont), jnp.asarray(x_mat)
        )
        vars_design = model.LensSystemVariables(
            curvature_parameters=curv,
            distances_z=dist,
            iors=iors,
        )
        return vars_design, np.asarray(material_ids_full, dtype=int)

    def _configure_plot_style(self, use_latex: bool = True) -> None:
        plt.style.use("default")
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.linestyle"] = (0, (5, 5))
        plt.rcParams["grid.linewidth"] = 0.5
        mpl.rcParams["font.size"] = 18
        plt.rcParams["xtick.labelsize"] = 18
        plt.rcParams["ytick.labelsize"] = 18

        if not use_latex:
            plt.rcParams["text.usetex"] = False
            return

        preamble_candidates = [
            Path("latex-preambula.tex"),
            Path(loss.__file__).resolve().parent.parent
            / "utils"
            / "latex-preambula.tex",
        ]
        preamble_path = next((x for x in preamble_candidates if x.exists()), None)
        if preamble_path is None:
            plt.rcParams["text.usetex"] = False
            return

        with open(preamble_path, "r") as f:
            latex_preamble = f.read()
        plt.rcParams["text.usetex"] = True
        plt.rc("text.latex", preamble=latex_preamble)

    def visualize(
        self,
        *,
        theta: Sequence[float] | None = None,
        vars_design: model.LensSystemVariables | None = None,
        material_ids: Sequence[int] | None = None,
        loss_value: float | None = None,
        use_latex: bool = True,
        num_rays: int = 10,
        field_factors: tuple = (-1, -0.7, 0, 0.7, 1),
        width_height_cm: tuple = (1.5 * 16, 1.5 * 9),
        is_add_aperture_stop: bool = True,
        is_colorbar: bool = False,
        is_post_optimized: bool = False,
        post_optimized_label: str = "Post-optimized design",
    ):
        if theta is not None:
            vars_design, material_ids = self.unpack_theta_to_design(theta)
            if loss_value is None:
                loss_value = self.objective_theta(theta)

        if vars_design is None or material_ids is None:
            raise ValueError(
                "Provide either `theta` or (`vars_design` and `material_ids`)."
            )

        if loss_value is None:
            loss_value = float(
                jnp.sum(
                    loss.loss(
                        self.cnst,
                        vars_design.curvature_parameters,
                        vars_design.distances_z,
                        vars_design.iors,
                    )
                )
            )

        self._configure_plot_style(use_latex=use_latex)

        def _rgb_to_color(r, g, b):
            return tuple(np.array([r, g, b]) / 255.0)

        colors = [
            _rgb_to_color(255, 217, 47),
            _rgb_to_color(231, 138, 195),
            _rgb_to_color(141, 160, 203),
            _rgb_to_color(166, 216, 84),
            _rgb_to_color(252, 141, 98),
        ]

        props: model.LensSystemComputedProperties = optics.compute_optical_properties(
            self.cnst, vars_design
        )

        fig, ax = lens_vis.plot_lenses_with_rays(
            cnst=self.cnst,
            vars=vars_design,
            material_ids=np.asarray(material_ids, dtype=int),
            sensor_z=props.sensor_z,
            num_rays=num_rays,
            field_factors=tuple(field_factors),
            colors=colors,
            is_add_aperture_stop=is_add_aperture_stop,
            width_hieght_cm=width_height_cm,
            is_colorbar=is_colorbar,
        )

        fig.text(0.5, 0.90, f"loss = {float(loss_value):.8f}", ha="center", va="bottom")
        if is_post_optimized:
            fig.text(
                0.98,
                0.96,
                post_optimized_label,
                ha="right",
                va="top",
                fontsize=14,
                fontweight="bold",
                color="darkgreen",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "darkgreen",
                    "alpha": 0.85,
                },
            )

        return fig, ax, float(loss_value)