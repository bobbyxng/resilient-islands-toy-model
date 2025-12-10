# Toy model for resilient islands with before/after capacity comparison
import logging
import numpy as np
import pandas as pd
import pypsa
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def extract_capacity(n: pypsa.Network) -> pd.DataFrame:
    """
    Extract a tidy table of optimal capacities for all extendable assets.
    """
    df = n.statistics.optimal_capacity()
    df = df.reset_index()  # component, name, capacity
    df.columns = ["component", "name", "capacity"]
    return df


def extract_energy_balance(n: pypsa.Network) -> pd.DataFrame:
    """
    Extract a tidy table of energy balance statistics.
    """
    df = n.statistics.energy_balance()
    df.rename("energy", inplace=True)
    df = df.reset_index()  # carrier, name, energy_in, energy_out, etc.
    return df


# Main
if __name__ == "__main__":

    # Settings
    n_elec_path = "data/networks/base_s_50_elec_.nc"
    solver_name = "gurobi"

    # Load network
    n = pypsa.Network(n_elec_path)

    n.carriers.loc["", "color"] = "#aaaaaa"

    n.add("Carrier", "load shedding", color="#aa0000")

    # Increase load
    n.loads_t.p_set *= 1.5

    # Add load-shedding generators
    spatial_nodes = n.buses.index.tolist()
    ls_names = [f"{bus} load shedding" for bus in spatial_nodes]

    n.add(
        "Generator",
        pd.Index(ls_names),
        bus=pd.Series(spatial_nodes, index=ls_names),
        p_nom_extendable=True,
        p_nom_max=np.inf,
        capital_cost=0.1,
        marginal_cost=10000.0,
        carrier="load shedding",
    )

    ### Run optimisations
    # Reference run
    print("Running baseline optimisation...")
    n1 = n.copy()
    n1.optimize(solver_name=solver_name)

    cap1 = extract_capacity(n1)
    eb1 = extract_energy_balance(n1)


    # S_max_pu = 0 run
    print("Running islanded optimisation (s_max_pu = 0 and p_max_pu = 0)...")
    n2 = n.copy()
    if not n.lines.empty:
        n2.lines["s_max_pu"] = 0
    if not n2.links.empty:
        dc_links = n2.links.index[n2.links.carrier == "DC"]
        n2.links.loc[dc_links, "p_max_pu"] = 0
        n2.links.loc[dc_links, "p_min_pu"] = 0

    n2.optimize(solver_name=solver_name)

    cap2 = extract_capacity(n2)
    eb2 = extract_energy_balance(n2)

    # Lines removed run
    print("Running islanded optimisation (lines and links removed)...")
    n3 = n.copy()
    if not n.lines.empty:
        n3.remove("Line", n3.lines.index)
    if not n3.links.empty:
        dc_links = n3.links.index[n3.links.carrier == "DC"]
        n3.remove("Link", dc_links)

    n3.optimize(solver_name=solver_name)

    cap3 = extract_capacity(n3)
    eb3 = extract_energy_balance(n3)

    ### Merge and compare results

    # Capacities
    capacities = []
    capacities.append(cap1.assign(scenario="grid"))
    capacities.append(cap2.assign(scenario="maxpu0"))
    capacities.append(cap3.assign(scenario="removed"))

    capacities = pd.concat(capacities, ignore_index=True)
    capacities_pivot = capacities.pivot_table(
        index=["component", "name"],
        columns="scenario",
        values="capacity",
        fill_value=0,
    ).reset_index()


    ### Plot capacities
    fig = go.Figure()

    fig.add_trace(go.Bar(x=capacities_pivot.name, y=capacities_pivot["grid"], name="Grid exists"))
    fig.add_trace(go.Bar(x=capacities_pivot.name, y=capacities_pivot["maxpu0"], name="Grid removed (s_max_pu=0)"))
    fig.add_trace(go.Bar(x=capacities_pivot.name, y=capacities_pivot["removed"], name="Grid removed (lines removed)"))

    fig.update_layout(
        title="Optimal capacity",
        xaxis_title="Carrier",
        yaxis_title="Installed Capacity [MW]",
        barmode="group",      # side-by-side comparison
        hovermode="x unified",
        template="plotly_white",
    )

    fig.show()

    # Energy balance
    energy_balances = []
    energy_balances.append(eb1.assign(scenario="grid"))
    energy_balances.append(eb2.assign(scenario="maxpu0"))
    energy_balances.append(eb3.assign(scenario="removed"))
    
    energy_balances = pd.concat(energy_balances, ignore_index=True)
    energy_balances_pivot = energy_balances.pivot_table(
        index="carrier",
        columns="scenario",
        values="energy",
        fill_value=0,
    ).reset_index()

    ### Plot energy balances
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=energy_balances_pivot.carrier, y=energy_balances_pivot["grid"], name="Grid exists"))
    fig2.add_trace(go.Bar(x=energy_balances_pivot.carrier, y=energy_balances_pivot["maxpu0"], name="Grid removed (s_max_pu=0)"))
    fig2.add_trace(go.Bar(x=energy_balances_pivot.carrier, y=energy_balances_pivot["removed"], name="Grid removed (lines removed)"))

    fig2.update_layout(
        title="Energy balance",
        xaxis_title="Carrier",
        yaxis_title="Energy [MWh]",
        barmode="group",      # side-by-side comparison
        hovermode="x unified",
        template="plotly_white",
    )

    # Print capital costs and opex
    costs1 = pd.concat([n1.statistics.capex(), n1.statistics.opex()], axis=1, keys=["capex", "opex"])
    costs1 = costs1.fillna(0).sum().sum()
    
    costs2 = pd.concat([n2.statistics.capex(), n2.statistics.opex()], axis=1, keys=["capex", "opex"])
    costs2 = costs2[~(
        (
            (costs2.index.get_level_values("component") == "Line") &
            (costs2.index.get_level_values("carrier") == "AC")) |
        (
            (costs2.index.get_level_values("component") == "Link") &
            (costs2.index.get_level_values("carrier") == "DC"))
    )]
    costs2 = costs2.fillna(0).sum().sum()

    costs3 = pd.concat([n3.statistics.capex(), n3.statistics.opex()], axis=1, keys=["capex", "opex"])
    costs3 = costs3[~(
        (
            (costs3.index.get_level_values("component") == "Line") &
            (costs3.index.get_level_values("carrier") == "AC")) |
        (
            (costs3.index.get_level_values("component") == "Link") &
            (costs3.index.get_level_values("carrier") == "DC"))
    )]
    costs3 = costs3.fillna(0).sum().sum()


    total_costs = pd.DataFrame({
        "scenario": ["grid", "maxpu0", "removed"],
        "total_cost": [
            np.round(costs1/1e9, 3),
            np.round(costs2/1e9, 3),
            np.round(costs3/1e9, 3),
        ]
    })
    print("\nTotal annual costs by scenario:")
    print(total_costs.to_string(index=False))
    
    # Marginal prices
    mp_diff = n3.buses_t.marginal_price - n2.buses_t.marginal_price
    mp_diff.abs().mean()
