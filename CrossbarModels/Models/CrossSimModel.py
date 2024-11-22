#
# Copyright 2017-2023 Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with
# Sandia Corporation, the U.S. Government retains certain rights in this software.
#
# See LICENSE for full license details
#

import numpy as xp

##############################

# This file implements Python compact circuit solvers to simulate the effects
# of undesired resistances in the array

##############################


def CrossSim_Solve(
    matrix,
    parasiticResistance,
    vector,
    Verr_th = 2e-5,
    hide_convergence_msg=0
):
    """Wrapper that is used to implement a convergence loop around the circuit solver.

    Each solver uses successive under-relaxation.

    If the circuit solver fails to find a solution, the relaxation parameter will
    be reduced until the solver converges, or a lower limit on the relaxation parameter
    is reached (returns a ValueError)
    """
    solved, retry = False, False
    input,output = xp.shape(matrix)
    gamma = min(0.9,50/(input+output)/parasiticResistance)

    while not solved:
        solved = True
        try:
            result = mvm_parasitics(
                vector,
                matrix.copy(),
                parasiticResistance,
                gamma,
                Verr_th
            )

        except RuntimeError:
            solved, retry = False, True
            gamma *= 0.8
            if gamma <= 1e-2:
                raise ValueError("Parasitic MVM solver failed to converge")
    if retry and not hide_convergence_msg:
        print(
            "CrossSim - Reduced MVM convergence parameter to: {:.5f}".format(
                gamma,
            ),
        )

    return result


def mvm_parasitics(vector, matrix, parasiticResistance, gamma, Verr_th):
    """Calculates the MVM result including parasitic resistance, for a non-interleaved array.

    vector : input vector
    matrix : normalized conductance matrix
    """
    # Parasitic resistance
    Rp_in = Rp_out = parasiticResistance

    Niters_max = 100

    # Initialize error and number of iterations
    Verr = 1e9
    Niters = 0

    # Initial estimate of device currents
    # Input seen at every element
    dV0 = xp.tile(vector, (matrix.shape[0], 1))
    Ires = matrix * dV0
    dV = dV0.copy()
    
    # Iteratively calculate parasitics and update device currents
    while Verr > Verr_th and Niters < Niters_max:
        # Calculate parasitic voltage drops
        Isum_col = xp.cumsum(Ires, 1)
        Isum_row = xp.cumsum(Ires[::-1], 0)[::-1]

        Vdrops_col = Rp_out * xp.cumsum(Isum_col[:, ::-1], 1)[:, ::-1]
        Vdrops_row = Rp_in * xp.cumsum(Isum_row, 0)
        Vpar = Vdrops_col + Vdrops_row

        # Calculate the error for the current estimate of memristor currents
        VerrMat = dV0 - Vpar - dV

        # Evaluate overall error; if using SIMD, make sure only to count the cells that matter
        Verr = xp.max(xp.abs(VerrMat))
        if Verr < Verr_th:
            break

        # Update memristor currents for the next iteration
        dV += gamma * VerrMat
        Ires = matrix * dV
        Niters += 1

    # Calculate the summed currents on the columns
    Icols = xp.sum(Ires, axis=1)
    # Should add some more checks here on whether the results of this calculation are erroneous even if it converged
    if Verr > Verr_th:
        raise RuntimeError("Parasitic resistance too high: could not converge!")
    if xp.isnan(Icols).any():
        raise RuntimeError("Nans due to parasitic resistance simulation")
    
    return dV.T,Icols