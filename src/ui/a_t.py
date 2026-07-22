import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $a_t$ profile playground

    Guidance profiles on $[0,1]$ over the flow steps $t = 0,\dots,T-1$.
    Four modes, one shared parameter $\eta$ (used differently per mode), a floor
    $\varepsilon$ (the "small value"), and a **specular** twin per mode:
    time reversal for the monotone modes, vertical flip for the (time-symmetric)
    gaussian.
    """)
    return


@app.cell
def _(np):
    A_T_MODES = ["gaussian", "linear", "logistic", "gap-closing"]

    def a_t_profile(mode, eta, T, floor=0.01, specular=False):
        """Guidance profile a_t in [floor, 1] over t = 0..T-1.

        eta in (0, 1] means, per mode:
          gaussian    end level E = floor + eta*(1-floor): variance spread -- the
                      bell's tails 'leave the box' as eta grows
          linear      inclination: finish F = 1 - eta*(1-floor) (eta=1 -> floor)
          logistic    same endpoints as linear (1 -> F), S-shaped in between
          gap-closing the remaining-gap schedule (1-eta)^(t+1)

        specular: time reversal (linear/logistic/gap-closing); the gaussian is
        time-symmetric, so its twin is the VERTICAL flip (valley at the middle).
        """
        eta = float(np.clip(eta, 1e-6, 1.0))
        floor = float(np.clip(floor, 0.0, 0.99))
        x = np.linspace(0.0, 1.0, T)

        if mode == "gaussian":
            end = min(floor + eta * (1.0 - floor), 0.999)
            sigma = 0.5 / np.sqrt(2.0 * np.log(1.0 / end))
            a = np.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
            if specular:
                a = 1.0 + end - a  # vertical flip: ends at 1, valley at `end`
        elif mode == "linear":
            finish = 1.0 - eta * (1.0 - floor)
            a = 1.0 + (finish - 1.0) * x
        elif mode == "logistic":
            # same endpoints as linear, S-shaped; affinely normalized so the raw
            # logistic hits the endpoints exactly despite its asymptotic tails
            finish = 1.0 - eta * (1.0 - floor)
            k = 10.0
            raw = 1.0 / (1.0 + np.exp(-k * (0.5 - x)))
            raw = (raw - raw[-1]) / (raw[0] - raw[-1])
            a = finish + (1.0 - finish) * raw
        elif mode == "gap-closing":
            a = (1.0 - eta) ** (np.arange(T) + 1)
        else:
            raise ValueError(f"unknown a_t mode {mode!r}")

        if specular and mode != "gaussian":
            a = a[::-1]
        return np.clip(a, floor, 1.0)
    return A_T_MODES, a_t_profile


@app.cell
def _(A_T_MODES, mo):
    mode_dropdown = mo.ui.dropdown(A_T_MODES, value="gaussian", label="mode: ")
    specular_checkbox = mo.ui.checkbox(label="specular twin")
    eta_slider = mo.ui.slider(0.01, 1.0, step=0.01, value=0.3, label="eta: ", show_value=True)
    floor_slider = mo.ui.slider(0.0, 0.2, step=0.01, value=0.01, label="floor: ", show_value=True)
    T_slider = mo.ui.slider(5, 50, step=1, value=25, label="T: ", show_value=True)
    mo.hstack([mode_dropdown, specular_checkbox, eta_slider, floor_slider, T_slider],
              justify="start", align="center")
    return T_slider, eta_slider, floor_slider, mode_dropdown, specular_checkbox


@app.cell
def _(T_slider, a_t_profile, eta_slider, floor_slider, mo, mode_dropdown, np, plt, specular_checkbox):
    _T = T_slider.value
    _a = a_t_profile(mode_dropdown.value, eta_slider.value, _T,
                     floor=floor_slider.value, specular=specular_checkbox.value)
    _twin = a_t_profile(mode_dropdown.value, eta_slider.value, _T,
                        floor=floor_slider.value, specular=not specular_checkbox.value)
    _t = np.arange(_T)

    _fig, _ax = plt.subplots(figsize=(9, 4), dpi=120)
    _ax.plot(_t, _a, "-o", color="#7B2CBF", markersize=4, label="selected")
    _ax.plot(_t, _twin, "--", color="#7B2CBF", alpha=0.35, linewidth=1.2, label="its twin")
    _ax.axhline(floor_slider.value, color="#BBBBBB", linewidth=0.8, linestyle=":")
    _ax.axhline(1.0, color="#BBBBBB", linewidth=0.8, linestyle=":")
    _ax.set_ylim(-0.03, 1.08)
    _ax.set_xlabel("$t$"); _ax.set_ylabel("$a_t$")
    _ax.set_title(
        f"{mode_dropdown.value}{' (specular)' if specular_checkbox.value else ''}"
        rf"  —  $\eta$={eta_slider.value:g}, $\varepsilon$={floor_slider.value:g}",
        loc="left",
    )
    for _sp in ("top", "right"):
        _ax.spines[_sp].set_visible(False)
    _ax.legend(frameon=False, fontsize=8)
    _ax.yaxis.grid(True, color="#EEEEEE")
    mo.mpl.interactive(_fig) if False else _fig
    return


@app.cell
def _(A_T_MODES, T_slider, a_t_profile, eta_slider, floor_slider, np, plt):
    # small multiples: every mode x (normal, specular) at the current eta / floor / T
    _T = T_slider.value
    _t = np.arange(_T)
    _fig, _axes = plt.subplots(2, len(A_T_MODES), figsize=(3.2 * len(A_T_MODES), 5),
                               dpi=120, sharex=True, sharey=True)
    for _j, _mode in enumerate(A_T_MODES):
        for _i, _spec in enumerate((False, True)):
            _ax = _axes[_i, _j]
            _a = a_t_profile(_mode, eta_slider.value, _T,
                             floor=floor_slider.value, specular=_spec)
            _ax.plot(_t, _a, "-o", color="#7B2CBF", markersize=2.5, linewidth=1.2)
            _ax.axhline(floor_slider.value, color="#CCCCCC", linewidth=0.6, linestyle=":")
            _ax.set_ylim(-0.03, 1.08)
            _ax.set_title(f"{_mode}{' (spec)' if _spec else ''}", fontsize=9)
            for _sp in ("top", "right"):
                _ax.spines[_sp].set_visible(False)
    _fig.supxlabel("$t$"); _fig.supylabel("$a_t$")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
