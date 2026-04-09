# tricks

- run scripts inside folder faking them to be outside: 
    ```
    uv run -m experiments.rundels/.venv/lib/python3.11/site-packages/timm/models/layers/__init__.py:49:
    ```
- (use svgs rather than pngs) to set dpi of all figures with:
    ```    
    # import matplotlib as mpl
    # mpl.rcParams["figure.dpi"] = 500
    ```
- lambda funcs:
    ```
    tensordict_apply(lambda z, u: z + dt * u, z_t, u_t)
    # the lambda corresponds to def f(z, u): return z + h * u, the rest are the arguments
    ```