from firedrake.preconditioners.pmg import PMGPC, PMGSNES

__all__ = ("P1PC", "P1SNES", "LORPC")


class P1PC(PMGPC):
    """Apply a two-level preconditioner with aggressive polynomial coarsening.

    Coarsen directly to the requested coarse degree.

    Notes
    -----
    .. rubric:: PETSc options

    The keys below are relative to the outer solver options prefix.

    ``pmg_mg_coarse_degree``, ``pmg_mg_coarse_mat_type``,
    ``pmg_mg_coarse_pmat_type``, ``pmg_mg_coarse_form_compiler_mode``,
    and ``pmg_mg_levels_transfer_mat_type`` inherit their types, defaults,
    and meanings from :class:`~.PMGBase`. Inner relaxation and coarse solves
    use ``pmg_mg_levels_`` and ``pmg_mg_coarse_``.

    ``pmg_mg_coarse_pc_type`` and ``pmg_mg_coarse_pc_mg_levels`` inherit
    their defaults and conditional geometric-level cap from :class:`~.PMGPC`.
    """
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        return ele.reconstruct(degree=self.coarse_degree)


class P1SNES(PMGSNES):
    """Apply a two-level nonlinear solver with aggressive polynomial coarsening.

    Coarsen directly to the requested coarse degree.

    Notes
    -----
    .. rubric:: PETSc options

    The keys below are relative to the outer solver options prefix.

    ``pfas_fas_coarse_degree``, ``pfas_fas_coarse_mat_type``,
    ``pfas_fas_coarse_pmat_type``, ``pfas_fas_coarse_form_compiler_mode``,
    and ``pfas_mg_levels_transfer_mat_type`` inherit their types, defaults,
    and meanings from :class:`~.PMGBase`. Inner relaxation and coarse solves
    use ``pfas_fas_levels_`` and ``pfas_fas_coarse_``.

    ``pfas_fas_coarse_pc_type``, ``pfas_fas_coarse_pc_mg_levels``,
    ``pfas_fas_coarse_snes_type``, and ``pfas_fas_coarse_snes_fas_levels``
    inherit their defaults and conditional geometric-level caps from
    :class:`~.PMGSNES`.
    """
    def coarsen_element(self, ele):
        if super().max_degree(ele) <= self.coarse_degree:
            raise ValueError
        return ele.reconstruct(degree=self.coarse_degree)


class LORPC(PMGPC):
    """Apply a low-order refined preconditioner with a P1-iso-Pk coarse space.

    Coarsen directly to the requested coarse degree.

    Notes
    -----
    .. rubric:: PETSc options

    The keys below are relative to the outer solver options prefix.

    ``lor_mg_coarse_degree``, ``lor_mg_coarse_mat_type``,
    ``lor_mg_coarse_pmat_type``, ``lor_mg_coarse_form_compiler_mode``,
    and ``lor_mg_levels_transfer_mat_type`` inherit their types, defaults,
    and meanings from :class:`~.PMGBase`. Inner relaxation and coarse solves
    use ``lor_mg_levels_`` and ``lor_mg_coarse_``.

    ``pmg_mg_coarse_pc_type`` and ``pmg_mg_coarse_pc_mg_levels`` inherit
    their defaults and conditional geometric-level cap from :class:`~.PMGPC`. These
    inherited checks retain the literal ``pmg_`` prefix even though the
    internal solver uses ``lor_``.
    """

    _prefix = "lor_"

    def coarsen_element(self, ele):
        degree = super().max_degree(ele)
        if degree <= self.coarse_degree:
            raise ValueError
        variant = ele.variant()
        if variant is None:
            iso_variant = f"iso({degree})"
        else:
            iso_variant = f"{variant},iso({degree})"
        cele = ele.reconstruct(degree=self.coarse_degree)
        cele = cele.reconstruct(variant=iso_variant)
        return cele
