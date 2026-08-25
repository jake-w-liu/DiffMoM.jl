#!/usr/bin/env python3
"""Run open-source periodic cross-validation in Meep using Julia-exported geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from typing import Any, Dict, List

from _meep_common import (
    add_project_root_argument,
    add_runtime_arguments,
    load_json_object,
    validate_runtime_geometry,
)


def build_meep_geometry(
    mp: Any, mask: List[List[int]], dx_cell: float, dy_cell: float, thickness: float
) -> List[Any]:
    ny = len(mask)
    nx = len(mask[0]) if ny else 0
    if nx == 0:
        return []

    dx_pix = dx_cell / nx
    dy_pix = dy_cell / ny
    geometry = []

    for jy, row in enumerate(mask):
        y = -0.5 * dy_cell + (jy + 0.5) * dy_pix
        for jx, is_metal in enumerate(row):
            if int(is_metal) != 1:
                continue
            x = -0.5 * dx_cell + (jx + 0.5) * dx_pix
            geometry.append(
                mp.Block(
                    size=mp.Vector3(dx_pix, dy_pix, thickness),
                    center=mp.Vector3(x, y, 0.0),
                    material=mp.metal,
                )
            )
    return geometry


def run_meep_case(
    geometry_json: Dict[str, Any],
    resolution: int,
    pml_lambda: float,
    sz_lambda: float,
    metal_thickness_lambda: float,
    source_offset_lambda: float,
    refl_offset_lambda: float,
    tran_offset_lambda: float,
    fwidth: float,
    after_sources_time: float,
) -> Dict[str, Any]:
    try:
        import meep as mp
    except ImportError as exc:
        raise SystemExit(
            f"Could not import Meep or one of its dependencies: {exc}. Run "
            "conda install -c conda-forge pymeep, then rerun the command."
        ) from exc

    dx_lambda = float(geometry_json["dx_lambda"])
    dy_lambda = float(geometry_json["dy_lambda"])
    mask = geometry_json["metal_mask_row_major"]

    fcen = 1.0  # normalized by lambda
    cell_size = mp.Vector3(dx_lambda, dy_lambda, sz_lambda)
    boundary_layers = [mp.PML(pml_lambda, direction=mp.Z)]

    src_z = 0.5 * sz_lambda - pml_lambda - source_offset_lambda
    refl_z = src_z - refl_offset_lambda
    tran_z = -0.5 * sz_lambda + pml_lambda + tran_offset_lambda
    if refl_z <= tran_z:
        raise SystemExit(
            "Flux monitor placement invalid: reflection plane must be above transmission plane."
        )

    source = mp.Source(
        src=mp.GaussianSource(fcen, fwidth=fwidth),
        component=mp.Ex,
        center=mp.Vector3(0.0, 0.0, src_z),
        size=mp.Vector3(dx_lambda, dy_lambda, 0.0),
        amplitude=1.0,
    )
    sources = [source]

    refl_region = mp.FluxRegion(
        center=mp.Vector3(0.0, 0.0, refl_z),
        size=mp.Vector3(dx_lambda, dy_lambda, 0.0),
    )
    tran_region = mp.FluxRegion(
        center=mp.Vector3(0.0, 0.0, tran_z),
        size=mp.Vector3(dx_lambda, dy_lambda, 0.0),
    )

    sim_empty = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        k_point=mp.Vector3(),
        sources=sources,
        resolution=resolution,
    )
    refl_empty = sim_empty.add_flux(fcen, 0.0, 1, refl_region)
    tran_empty = sim_empty.add_flux(fcen, 0.0, 1, tran_region)
    sim_empty.run(until_after_sources=after_sources_time)

    incident_flux = float(mp.get_fluxes(tran_empty)[0])
    incident_refl_data = sim_empty.get_flux_data(refl_empty)
    incident_refl_flux = float(mp.get_fluxes(refl_empty)[0])
    sim_empty.reset_meep()

    metal_thickness = max(metal_thickness_lambda, 1.0 / resolution)
    geometry = build_meep_geometry(mp, mask, dx_lambda, dy_lambda, metal_thickness)

    sim_geom = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        k_point=mp.Vector3(),
        sources=sources,
        geometry=geometry,
        resolution=resolution,
    )
    refl = sim_geom.add_flux(fcen, 0.0, 1, refl_region)
    tran = sim_geom.add_flux(fcen, 0.0, 1, tran_region)
    sim_geom.load_minus_flux_data(refl, incident_refl_data)
    sim_geom.run(until_after_sources=after_sources_time)

    refl_flux = float(mp.get_fluxes(refl)[0])
    tran_flux = float(mp.get_fluxes(tran)[0])
    sim_geom.reset_meep()

    if not all(
        math.isfinite(value)
        for value in (incident_flux, incident_refl_flux, refl_flux, tran_flux)
    ):
        raise SystemExit(
            "Meep returned a non-finite flux value. Check the resolution, source, "
            "monitor placement, and simulation duration before rerunning."
        )
    if abs(incident_flux) < 1e-15:
        raise SystemExit(
            "Incident normalization flux is near zero. Adjust the source or monitor "
            "placement before rerunning."
        )

    reflectance = max(0.0, -refl_flux / incident_flux)
    transmittance = max(0.0, tran_flux / incident_flux)
    absorption = 1.0 - reflectance - transmittance
    if not all(
        math.isfinite(value) for value in (reflectance, transmittance, absorption)
    ):
        raise SystemExit(
            "Computed Meep power totals contain a non-finite value. Check the flux "
            "normalization and rerun the simulation."
        )

    return {
        "meep_frequency_normalized": fcen,
        "resolution_px_per_lambda": resolution,
        "pml_lambda": pml_lambda,
        "sz_lambda": sz_lambda,
        "metal_thickness_lambda": metal_thickness,
        "source_z_lambda": src_z,
        "refl_z_lambda": refl_z,
        "tran_z_lambda": tran_z,
        "fwidth": fwidth,
        "after_sources_time": after_sources_time,
        "num_metal_blocks": len(geometry),
        "incident_flux": incident_flux,
        "incident_refl_flux_empty": incident_refl_flux,
        "reflected_flux": refl_flux,
        "transmitted_flux": tran_flux,
        "reflectance_total": reflectance,
        "transmittance_total": transmittance,
        "absorption_total": absorption,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_project_root_argument(parser, __file__)
    parser.add_argument("--output-prefix", type=str, default="meep_periodic")
    add_runtime_arguments(
        parser, resolution_default=36, after_sources_default=200.0
    )
    args = parser.parse_args()
    validate_runtime_geometry(parser, args)

    data_dir = args.project_root / "data"
    geometry_path = data_dir / f"julia_{args.output_prefix}_geometry.json"
    reference_path = data_dir / f"julia_{args.output_prefix}_reference.json"
    results_json_path = data_dir / f"meep_{args.output_prefix}_results.json"
    results_csv_path = data_dir / f"meep_{args.output_prefix}_results.csv"

    if not geometry_path.exists():
        raise SystemExit(
            f"Missing Julia geometry file: {geometry_path}. Run `julia --project=. "
            "validation/meep/run_periodic_case_julia_reference.jl` with the same "
            "output prefix, then rerun Meep."
        )
    if not reference_path.exists():
        raise SystemExit(
            f"Missing Julia reference file: {reference_path}. Run `julia --project=. "
            "validation/meep/run_periodic_case_julia_reference.jl` with the same "
            "output prefix, then rerun Meep."
        )

    recovery = "Regenerate the Julia artifact before running Meep."
    geometry_json = load_json_object(geometry_path, recovery=recovery)
    reference_json = load_json_object(reference_path, recovery=recovery)

    meep_results = run_meep_case(
        geometry_json=geometry_json,
        resolution=args.resolution,
        pml_lambda=args.pml_lambda,
        sz_lambda=args.sz_lambda,
        metal_thickness_lambda=args.metal_thickness_lambda,
        source_offset_lambda=args.source_offset_lambda,
        refl_offset_lambda=args.refl_offset_lambda,
        tran_offset_lambda=args.tran_offset_lambda,
        fwidth=args.fwidth,
        after_sources_time=args.after_sources_time,
    )

    payload = {
        "output_prefix": args.output_prefix,
        "frequency_ghz": reference_json["frequency_ghz"],
        "lambda_m": reference_json["lambda_m"],
        "dx_lambda": geometry_json["dx_lambda"],
        "dy_lambda": geometry_json["dy_lambda"],
        "nx": geometry_json["nx"],
        "ny": geometry_json["ny"],
        "metal_fill_fraction": geometry_json["metal_fill_fraction"],
        **meep_results,
    }

    with results_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")

    with results_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)

    print(f"Saved {results_json_path}")
    print(f"Saved {results_csv_path}")
    print(
        "Meep totals: "
        f"R={payload['reflectance_total']:.6f}, "
        f"T={payload['transmittance_total']:.6f}, "
        f"A={payload['absorption_total']:.6f}"
    )


if __name__ == "__main__":
    main()
