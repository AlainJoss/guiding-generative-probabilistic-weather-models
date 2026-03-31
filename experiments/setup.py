import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    x_start = ...
    return


@app.cell
def _(ensemble_rollouts):
    from src.utils import ensure_dir
    result_dir = ensure_dir(ensemble_rollouts)
    return


app._unparsable_cell(
    r"""
    def ensemble_rollout(result_dir: str, M: int, N: int, x_start):
        for m in range(M):
            for n in range(N):
            
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup experiment
    - Define N
    - Define M: number of members in ensemble
    - Define TIMESTAMP: starting datetime
    - Retrieve x_start from TIMESTAMP
    - -> Rollout ensemble and collect trajectories

    In next notebook:
    - Define mask
    - Represent average change over mask in time (ensemble trajectories)
    - Define "extrimified trajectory"
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
