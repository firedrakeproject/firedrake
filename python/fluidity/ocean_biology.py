def pznd(state, parameters):
    '''Calculate sources and sinks for a simple PZND model'''

    if not check_pznd_parameters(parameters):
        raise TypeError("Missing Parameter")

    P = state.scalar_fields["Phytoplankton"]
    Z = state.scalar_fields["Zooplankton"]
    N = state.scalar_fields["Nutrient"]
    D = state.scalar_fields["Detritus"]
    # Note: *NOT* PhotosyntheticRadation field, but the internal _PAR field,
    # which
    I = state.scalar_fields["_PAR"]
    # has been projected onto the same mesh as phytoplankton and has been
    # converted from incident radiation to just the active part.
    Pnew = state.scalar_fields["IteratedPhytoplankton"]
    Znew = state.scalar_fields["IteratedZooplankton"]
    Nnew = state.scalar_fields["IteratedNutrient"]
    Dnew = state.scalar_fields["IteratedDetritus"]

    P_source = state.scalar_fields["PhytoplanktonSource"]
    Z_source = state.scalar_fields["ZooplanktonSource"]
    N_source = state.scalar_fields["NutrientSource"]
    D_source = state.scalar_fields["DetritusSource"]
    try:
        PP = state.scalar_fields["PrimaryProduction"]
    except KeyError:
        PP = None
    try:
        PG = state.scalar_fields["PhytoplanktonGrazing"]
    except KeyError:
        PG = None

    alpha = parameters["alpha"]
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    g = parameters["g"]
    k_N = parameters["k_N"]
    k = parameters["k"]
    v = parameters["v"]
    mu_P = parameters["mu_P"]
    mu_Z = parameters["mu_Z"]
    mu_D = parameters["mu_D"]
    p_P = parameters["p_P"]
    p_D = 1 - p_P

    for n in range(P.node_count):
        # Values of fields on this node.
        P_n = max(.5 * (P.node_val(n) + Pnew.node_val(n)), 0.0)
        Z_n = max(.5 * (Z.node_val(n) + Znew.node_val(n)), 0.0)
        N_n = max(.5 * (N.node_val(n) + Nnew.node_val(n)), 0.0)
        D_n = max(.5 * (D.node_val(n) + Dnew.node_val(n)), 0.0)
        I_n = max(I.node_val(n), 0.0)

        if (I_n < 0.0001):
            I_n = 0.0

        # Light limited phytoplankton growth rate.
        J = (v * alpha * I_n) / (v ** 2 + alpha ** 2 * I_n ** 2) ** 0.5

        # Nitrate limiting factor.
        Q = N_n / (k_N + N_n)

        # Total phytoplankton growth rate.
        R_P = J * P_n * Q

        # Zooplankton grazing of phytoplankton.
        G_P = (g * p_P * P_n ** 2 * Z_n) / \
            (k ** 2 + p_P * P_n ** 2 + p_D * D_n ** 2)

        # Zooplankton grazing of detritus.
        G_D = (g * (1 - p_P) * D_n ** 2 * Z_n) / \
            (k ** 2 + p_P * P_n ** 2 + p_D * D_n ** 2)

        # Death rate of phytoplankton.
        De_P = mu_P * P_n * P_n / (P_n + 0.2)

        # Death rate of zooplankton.
        De_Z = mu_Z * Z_n * Z_n * Z_n / (Z_n + 3)

        # Detritus remineralisation.
        De_D = mu_D * D_n

        P_source.set(n, R_P - G_P - De_P)

        if PP:
            PP.set(n, R_P)
        if PG:
            PG.set(n, G_P)

        Z_source.set(n, gamma * beta * (G_P + G_D) - De_Z)

        N_source.set(n, -R_P + De_D + (1 - gamma) * beta * (G_P + G_D))

        D_source.set(n, -De_D + De_P + De_Z + (1 - beta) * G_P - beta * G_D)


def check_pznd_parameters(parameters):
    from sys import stderr

    valid = True

    if not "alpha" in parameters:
        stderr.write("PZND parameter alpha missing.\n")
        stderr.write("alpha is this initial slope of the P-I curve.\n\n")
        valid = False

    if not "beta" in parameters:
        stderr.write("PZND parameter beta missing.\n")
        stderr.write("beta is the assimilation efficiency of zooplankton.\n\n")
        valid = False

    if not "gamma" in parameters:
        stderr.write("PZND parameter gamma missing.\n")
        stderr.write("gamma is the zooplankton excretion parameter.\n\n")
        valid = False

    if not "g" in parameters:
        stderr.write("PZND parameter g missing.\n")
        stderr.write("g is the zooplankton maximum growth rate.\n\n")
        valid = False

    if not "k_N" in parameters:
        stderr.write("PZND parameter k_N missing.\n")
        stderr.write("k_N is the half-saturation constant for nutrient.\n\n")
        valid = False

    if not "k" in parameters:
        stderr.write("PZND parameter k missing.\n")
        stderr.write("k is the zooplankton grazing parameter.\n\n")
        valid = False

    if not "mu_P" in parameters:
        stderr.write("PZND parameter mu_P missing.\n")
        stderr.write("mu_P is the phytoplankton mortality rate.\n\n")
        valid = False

    if not "mu_Z" in parameters:
        stderr.write("PZND parameter mu_Z missing.\n")
        stderr.write("mu_Z is the zooplankton mortality rate.\n\n")
        valid = False

    if not "mu_D" in parameters:
        stderr.write("PZND parameter mu_D missing.\n")
        stderr.write("mu_D is the detritus remineralisation rate.\n\n")
        valid = False

    if not "p_P" in parameters:
        stderr.write("PZND parameter p_P missing.\n")
        stderr.write(
            "p_P is the relative grazing preference of zooplankton for phytoplankton.\n\n")
        valid = False

    if not "v" in parameters:
        stderr.write("PZND parameter v missing.\n")
        stderr.write("v is the maximum phytoplankton growth rate.\n\n")
        valid = False

    return valid


def lotka_volterra(state, parameters):

    if not check_lotka_volterra_parameters(parameters):
        raise TypeError("Missing Parameter")

    P = state.scalar_fields["Phytoplankton"]
    Z = state.scalar_fields["Zooplankton"]
    Pnew = state.scalar_fields["IteratedPhytoplankton"]
    Znew = state.scalar_fields["IteratedZooplankton"]

    P_source = state.scalar_fields["PhytoplanktonSource"]
    Z_source = state.scalar_fields["ZooplanktonSource"]

    alpha = parameters["alpha"]
    beta = parameters["beta"]
    gamma = parameters["gamma"]
    delta = parameters["delta"]

    for n in range(P.node_count):
        # Values of fields on this node.
        P_n = .5 * (P.node_val(n) + Pnew.node_val(n))
        Z_n = .5 * (Z.node_val(n) + Znew.node_val(n))

        P_source.set(n, P_n * (alpha - beta * Z_n))

        Z_source.set(n, -Z_n * (gamma - delta * P_n))


def check_lotka_volterra_parameters(parameters):
    from sys import stderr

    valid = True

    if not "alpha" in parameters:
        stderr.write("Lotka Voltera parameter alpha missing.\n")
        valid = False

    if not "beta" in parameters:
        stderr.write("Lotka Voltera parameter beta missing.\n")
        valid = False

    if not "gamma" in parameters:
        stderr.write("Lotka Voltera parameter gamma missing.\n")
        valid = False

    if not "delta" in parameters:
        stderr.write("Lotka Voltera parameter delta missing.\n")
        valid = False

    if not valid:
        stderr.write(" dP/dt = P*(alpha-beta * Z)")
        stderr.write(" dZ/dt = - Z*(gamma-delta * P)")

    return valid

#
#
# pczdna model                         #
#
#


def six_component(state, parameters):
    '''Calculate sources and sinks for pczdna biology model'''

    # Based on the equations in
    # Popova, E. E.; Coward, A. C.; Nurser, G. A.; de Cuevas, B.;
    # Fasham, M. J. R. & Anderson, T. R.
    # Mechanisms controlling primary and new production in a global ecosystem
    # model - Part I:
    # Validation of the biological simulation Ocean Science, 2006, 2, 249-266.
    # DOI: 10.5194/os-2-249-2006
    import math

    if not check_six_component_parameters(parameters):
        raise TypeError("Missing Parameter")

    P = state.scalar_fields["Phytoplankton"]
    C = state.scalar_fields["Chlorophyll"]
    Z = state.scalar_fields["Zooplankton"]
    N = state.scalar_fields["Nutrient"]
    A = state.scalar_fields["Ammonium"]
    D = state.scalar_fields["Detritus"]
    # Note: *NOT* PhotosyntheticRadation field, but the internal _PAR field,
    # which
    I = state.scalar_fields["_PAR"]
    # has been projected onto the same mesh as phytoplankton and has been
    # converted from incident radiation to just the active part.
    Pnew = state.scalar_fields["IteratedPhytoplankton"]
    Cnew = state.scalar_fields["IteratedChlorophyll"]
    Znew = state.scalar_fields["IteratedZooplankton"]
    Nnew = state.scalar_fields["IteratedNutrient"]
    Anew = state.scalar_fields["IteratedAmmonium"]
    Dnew = state.scalar_fields["IteratedDetritus"]
    coords = state.vector_fields["Coordinate"]

    P_source = state.scalar_fields["PhytoplanktonSource"]
    C_source = state.scalar_fields["ChlorophyllSource"]
    Z_source = state.scalar_fields["ZooplanktonSource"]
    N_source = state.scalar_fields["NutrientSource"]
    A_source = state.scalar_fields["AmmoniumSource"]
    D_source = state.scalar_fields["DetritusSource"]
    try:
        PP = state.scalar_fields["PrimaryProduction"]
    except KeyError:
        PP = None
    try:
        PG = state.scalar_fields["PhytoplanktonGrazing"]
    except KeyError:
        PG = None

    alpha_c = parameters["alpha_c"]
    beta_P = parameters["beta_p"]
    beta_D = parameters["beta_d"]
    delta = parameters["delta"]
    gamma = parameters["gamma"]
    zeta = parameters["zeta"]
    epsilon = parameters["epsilon"]
    psi = parameters["psi"]
    g = parameters["g"]
    k_N = parameters["k_N"]
    k_A = parameters["k_A"]
    k_p = parameters["k_p"]
    k_z = parameters["k_z"]
    v = parameters["v"]
    mu_P = parameters["mu_P"]
    mu_Z = parameters["mu_Z"]
    mu_D = parameters["mu_D"]
    p_P = parameters["p_P"]
    theta_m = parameters["theta_m"]
    lambda_bio = parameters["lambda_bio"]
    lambda_A = parameters["lambda_A"]
    p_D = 1 - p_P

    for n in range(P.node_count):
        # Values of fields on this node.
        P_n = max(.5 * (P.node_val(n) + Pnew.node_val(n)), 0.0)
        Z_n = max(.5 * (Z.node_val(n) + Znew.node_val(n)), 0.0)
        N_n = max(.5 * (N.node_val(n) + Nnew.node_val(n)), 0.0)
        A_n = max(.5 * (A.node_val(n) + Anew.node_val(n)), 0.0)
        C_n = max(.5 * (C.node_val(n) + Cnew.node_val(n)), 0.0)
        D_n = max(.5 * (D.node_val(n) + Dnew.node_val(n)), 0.0)
        I_n = max(I.node_val(n), 0.0)
        depth = abs(coords.node_val(n)[2])

        if (I_n < 0.0001):
            I_n = 0

        # In the continuous model we start calculating Chl-a related
        # properties at light levels close to zero with a potential /0.
        # It seems that assuming theta = zeta at very low P and Chl takes
        # care of this most effectively
        if (P_n < 1e-7 or C_n < 1e-7):
            theta = zeta
        else:
            theta = C_n / P_n * zeta  # C=P_n*zeta
        alpha = alpha_c * theta

        # Light limited phytoplankton growth rate.
        J = (v * alpha * I_n) / (v ** 2 + alpha ** 2 * I_n ** 2) ** 0.5

        # Nitrate limiting factor.
        Q_N = (N_n * math.exp(-psi * A_n)) / (k_N + N_n)

        # Ammonium limiting factor
        Q_A = A_n / (k_A + A_n)

        # Chl growth scaling factor
        # R_P=(theta_m/theta)*J*(Q_N+Q_A)/(alpha*I_n+1e-7)
        R_P = (theta_m / theta) * (Q_N + Q_A) * \
            v / (v ** 2 + alpha ** 2 * I_n ** 2) ** 0.5

        # Primary production
        X_P = J * (Q_N + Q_A) * P_n

        # Zooplankton grazing of phytoplankton.
            # It looks a bit different from the original version, however
            # it is the same function with differently normalised parameters to
            # simplify tuning
        # G_P=(g * epsilon * p_P * P_n**2 * Z_n)/(g+epsilon*(p_P*P_n**2 + p_D*D_n**2))
        G_P = (g * p_P * P_n ** 2 * Z_n) / \
            (epsilon + (p_P * P_n ** 2 + p_D * D_n ** 2))

        # Zooplankton grazing of detritus. (p_D - 1-p_P)
        # G_D=(g * epsilon * (1-p_P) * D_n**2 * Z_n)/(g+epsilon*(p_P*P_n**2 + p_D*D_n**2))
        G_D = (g * (1 - p_P) * D_n ** 2 * Z_n) / \
            (epsilon + (p_P * P_n ** 2 + p_D * D_n ** 2))

        # Death rate of phytoplankton.
        # There is an additional linear term because we have a unified model
        # (no below/above photoc zone distinction)
        De_P = mu_P * P_n * P_n / (P_n + k_p) + lambda_bio * P_n

        # Death rate of zooplankton.
        # There is an additional linear term because we have a unified model
        # (no below/above photoc zone distinction)
        De_Z = mu_Z * Z_n ** 3 / (Z_n + k_z) + lambda_bio * Z_n

        # Detritus remineralisation.
        De_D = mu_D * D_n + lambda_bio * P_n + lambda_bio * Z_n

        # Ammonium nitrification (only below the photic zone)
            # This is the only above/below term
        De_A = lambda_A * A_n * (1 - photic_zone(depth, 100, 20))

        P_source.set(n, J * (Q_N + Q_A) * P_n - G_P - De_P)
        C_source.set(
            n, (R_P * J * (Q_N + Q_A) * P_n + (-G_P - De_P)) * theta / zeta)
        Z_source.set(n, delta * (beta_P * G_P + beta_D * G_D) - De_Z)
        D_source.set(
            n, -De_D + De_P + gamma * De_Z + (1 - beta_P) * G_P - beta_D * G_D)
        N_source.set(n, -J * P_n * Q_N + De_A)
        A_source.set(n, -J * P_n * Q_A + De_D + (1 - delta) * (
            beta_P * G_P + beta_D * G_D) + (1 - gamma) * De_Z - De_A)

        if PP:
            PP.set(n, X_P)
        if PG:
            PG.set(n, G_P)


def check_six_component_parameters(parameters):
    from sys import stderr

    valid = True

    if not "alpha_c" in parameters:
        stderr.write("PCZNDA parameter alpha_c missing.\n")
        stderr.write(
            "alpha is the chlorophyll-specific inital slope of P-I curve.\n\n")
        valid = False

    if not "beta_p" in parameters:
        stderr.write("PCZNDA parameter beta_p missing.\n")
        stderr.write(
            "beta is the assimilation efficiency of zooplankton for plankton.\n\n")
        valid = False

    if not "beta_d" in parameters:
        stderr.write("PCZNDA parameter beta_d missing.\n")
        stderr.write(
            "beta is the assimilation efficiency of zooplankton for detritus.\n\n")
        valid = False

    if not "delta" in parameters:
        stderr.write("PCZNDA parameter delta missing.\n")
        stderr.write("delta is the zooplankton excretion parameter.\n\n")
        valid = False

    if not "gamma" in parameters:
        stderr.write("PCZNDA parameter gamma missing.\n")
        stderr.write("gamma is the zooplankton excretion parameter.\n\n")
        valid = False

    if not "epsilon" in parameters:
        stderr.write("PCZNDA parameter epsilon missing.\n")
        stderr.write(
            "epsilon is the grazing parameter relating the rate of prey item to prey density.\n\n")
        valid = False

    if not "g" in parameters:
        stderr.write("PCZNDA parameter g missing.\n")
        stderr.write("g is the zooplankton maximum growth rate.\n\n")
        valid = False

    if not "k_A" in parameters:
        stderr.write("PCZNDA parameter k_A missing.\n")
        stderr.write("k_A is the half-saturation constant for ammonium.\n\n")
        valid = False

    if not "k_p" in parameters:
        stderr.write("PCZNDA parameter k_p missing.\n")
        stderr.write(
            "k_ is something to do with mortatility rate of phytoplankton")

    if not "k_z" in parameters:
        stderr.write("PCZNDA parameter k_z missing.\n")
        stderr.write(
            "k_z is something to do with te mortality rate of zooplankton\n\n")
        valid = False

    if not "k_N" in parameters:
        stderr.write("PCZNDA parameter k_N missing.\n")
        stderr.write("k_N is the half-saturation constant for nutrient.\n\n")
        valid = False

    if not "mu_P" in parameters:
        stderr.write("PCZNDA parameter mu_P missing.\n")
        stderr.write("mu_P is the phytoplankton mortality rate.\n\n")
        valid = False

    if not "mu_Z" in parameters:
        stderr.write("PCZNDA parameter mu_Z missing.\n")
        stderr.write("mu_Z is the zooplankton mortality rate.\n\n")
        valid = False

    if not "mu_D" in parameters:
        stderr.write("PCZNDA parameter mu_D missing.\n")
        stderr.write("mu_D is the detritus remineralisation rate.\n\n")
        valid = False

    if not "psi" in parameters:
        stderr.write("PCZNDA parameter psi missing.\n")
        stderr.write(
            "psi is the strength of ammonium inibition of nitrate uptake\n\n")
        valid = False

    if not "p_P" in parameters:
        stderr.write("PCZNDA parameter p_P missing.\n")
        stderr.write(
            "p_P is the relative grazing preference of zooplankton for phytoplankton.\n\n")
        valid = False

    if not "v" in parameters:
        stderr.write("PCZNDA parameter v missing.\n")
        stderr.write("v is the maximum phytoplankton growth rate.\n\n")
        valid = False

    if not "theta_m" in parameters:
        stderr.write("PCZNDA parameter theta_m missing.\n")
        stderr.write("theta_m is the maximum Chlorophyll to C ratio.\n\n")
        valid = False

    if not "zeta" in parameters:
        stderr.write("PCZNDA parameter zeta missing.\n")
        stderr.write(
            "zeta is the conversion factor from gC to mmolN on C:N ratio of 6.5\n\n")
        valid = False

    if not "lambda_bio" in parameters:
        stderr.write("PCZNDA parameter lambda_bio missing.\n")
        stderr.write(
            "lambda_bio is rate which plankton turn to detritus below photic zone\n\n")
        valid = False

    if not "lambda_A" in parameters:
        stderr.write("PCZNDA parameter lambda_A missing.\n")
        stderr.write("lambda_A nitrification rate below photic zone\n\n")
        valid = False

    if not "photic_zone_limit" in parameters:
        stderr.write("PCZNDA parameter photic_zone_limit missing.\n")
        stderr.write(
            "photic_zone_limit defines the base of the photic zone in W/m2\n\n")
        valid = False

    return valid


def photic_zone(z, limit, transition_length):

    depth = abs(z)
    if (depth < limit):
        return 1.
    elif (depth < limit + transition_length):
        return 1. - (depth - limit) / float(transition_length)
    else:
        return 0.0
