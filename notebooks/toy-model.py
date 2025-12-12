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
    df = n.statistics.optimal_capacity().reset_index()
    df.columns = ["component", "name", "capacity"]
    return df


def extract_energy_balance(n: pypsa.Network) -> pd.DataFrame:
    """
    Extract a tidy table of energy balance statistics.
    """
    df = n.statistics.energy_balance()
    df.rename("energy", inplace=True)
    return df.reset_index()


def calculate_total_costs(n: pypsa.Network, exclude_grid: bool = False) -> float:
    """
    Calculate total costs (capex + opex) for a network.
    If exclude_grid=True, excludes AC lines and DC links.
    """
    costs = pd.concat([n.statistics.capex(), n.statistics.opex()], axis=1, keys=["capex", "opex"])
    
    if exclude_grid:
        costs = costs[~(
            ((costs.index.get_level_values("component") == "Line") &
             (costs.index.get_level_values("carrier") == "AC")) |
            ((costs.index.get_level_values("component") == "Link") &
             (costs.index.get_level_values("carrier") == "DC"))
        )]
    
    return costs.fillna(0).sum().sum()


def create_bar_plot(pivot_df, x_col, y_cols, labels, title, yaxis_title):
    """
    Create a grouped bar plot with multiple scenarios.
    """
    fig = go.Figure()
    for y_col, label in zip(y_cols, labels):
        fig.add_trace(go.Bar(x=pivot_df[x_col], y=pivot_df[y_col], name=label))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col.title(),
        yaxis_title=yaxis_title,
        barmode="group",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


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
        dc_links = n2.links[n2.links.carrier == "DC"].index
        n2.links.loc[dc_links, ["p_max_pu", "p_min_pu"]] = 0

    n2.optimize(solver_name=solver_name)

    cap2 = extract_capacity(n2)
    eb2 = extract_energy_balance(n2)

    # Lines removed run
    print("Running islanded optimisation (lines and links removed)...")
    n3 = n.copy()
    if not n.lines.empty:
        n3.remove("Line", n3.lines.index)
    if not n3.links.empty:
        n3.remove("Link", n3.links[n3.links.carrier == "DC"].index)

    n3.optimize(solver_name=solver_name)

    cap3 = extract_capacity(n3)
    eb3 = extract_energy_balance(n3)

    ### Merge and compare results

    # Capacities
    capacities = pd.concat([cap1.assign(scenario="grid"), cap2.assign(scenario="maxpu0"), cap3.assign(scenario="removed")], ignore_index=True)
    capacities_pivot = capacities.pivot_table(
        index=["component", "name"],
        columns="scenario",
        values="capacity",
        fill_value=0,
    ).reset_index()

    # Plot capacities
    fig = create_bar_plot(
        capacities_pivot, "name",
        ["grid", "maxpu0", "removed"],
        ["Grid exists", "Grid removed (s_max_pu=0)", "Grid removed (lines removed)"],
        "Optimal capacity", "Installed Capacity [MW]"
    )
    fig.show()

    # Energy balance
    energy_balances = pd.concat([eb1.assign(scenario="grid"), eb2.assign(scenario="maxpu0"), eb3.assign(scenario="removed")], ignore_index=True)
    energy_balances_pivot = energy_balances.pivot_table(
        index="carrier",
        columns="scenario",
        values="energy",
        fill_value=0,
    ).reset_index()

    # Plot energy balances
    fig2 = create_bar_plot(
        energy_balances_pivot, "carrier",
        ["grid", "maxpu0", "removed"],
        ["Grid exists", "Grid removed (s_max_pu=0)", "Grid removed (lines removed)"],
        "Energy balance", "Energy [MWh]"
    )
    fig2.show()

    # Print capital costs and opex
    total_costs = pd.DataFrame({
        "scenario": ["grid", "maxpu0", "removed"],
        "total_cost": [
            np.round(calculate_total_costs(n1) / 1e9, 3),
            np.round(calculate_total_costs(n2, exclude_grid=True) / 1e9, 3),
            np.round(calculate_total_costs(n3, exclude_grid=True) / 1e9, 3),
        ]
    })
    print("\nTotal annual costs by scenario:")
    print(total_costs.to_string(index=False))
    
    # Marginal prices
    mp_diff = n3.buses_t.marginal_price - n2.buses_t.marginal_price
    mp_diff.abs().mean()
