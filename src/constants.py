PARTITIONS = ["surface", "level"]

LEVELS_DICT = {
    "surface": [0],
    "level": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
}

VARIABLES_DICT = {
    "surface": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure"
    ],
    "level": [
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "specific_humidity",
        "vertical_velocity"
    ]
}