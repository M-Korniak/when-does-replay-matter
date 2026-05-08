
HB_RANDOM_SCORE = {

    "h1-walk-v0": 2.377,

    "h1-stand-v0": 10.545,

    "h1-run-v0": 2.02,

    "h1-reach-v0": 260.302,

    "h1-hurdle-v0": 2.214,

    "h1-crawl-v0": 272.658,

    "h1-maze-v0": 106.441,

    "h1-sit_simple-v0": 9.393,

    "h1-sit_hard-v0": 2.448,

    "h1-balance_simple-v0": 9.391,

    "h1-balance_hard-v0": 9.044,

    "h1-stair-v0": 3.112,

    "h1-slide-v0": 3.191,

    "h1-pole-v0": 20.09,

    "h1hand-push-v0": -526.8,

    "h1hand-cabinet-v0": 37.733,

    "h1hand-door-v0": 2.771,

    "h1hand-truck-v0": 562.419,

    "h1hand-cube-v0": 4.787,

    "h1hand-bookshelf_simple-v0": 16.777,

    "h1hand-bookshelf_hard-v0": 14.848,

    "h1hand-basketball-v0": 8.979,

    "h1hand-window-v0": 2.713,

    "h1hand-spoon-v0": 4.661,

    "h1hand-kitchen-v0": 0.0,

    "h1hand-package-v0": -10040.932,

    "h1hand-powerlift-v0": 17.638,

    "h1hand-room-v0": 3.018,

    "h1hand-insert_small-v0": 1.653,

    "h1hand-insert_normal-v0": 1.673,

    "h1hand-walk-v0": 2.505,

    "h1hand-stand-v0": 11.973,

    "h1hand-run-v0": 1.927,

    "h1hand-reach-v0": -50.024,

    "h1hand-hurdle-v0": 2.371,

    "h1hand-crawl-v0": 278.868,

    "h1hand-maze-v0": 106.233,

    "h1hand-sit_simple-v0": 10.768,

    "h1hand-sit_hard-v0": 2.477,

    "h1hand-balance_simple-v0": 10.17,

    "h1hand-balance_hard-v0": 10.032,

    "h1hand-stair-v0": 3.161,

    "h1hand-slide-v0": 3.142,

    "h1hand-pole-v0": 19.721,

}



HB_SUCCESS_SCORE = {

    "h1-walk-v0": 700.0,

    "h1-stand-v0": 800.0,

    "h1-run-v0": 700.0,

    "h1-reach-v0": 12000.0,

    "h1-hurdle-v0": 700.0,

    "h1-crawl-v0": 700.0,

    "h1-maze-v0": 1200.0,

    "h1-sit_simple-v0": 750.0,

    "h1-sit_hard-v0": 750.0,

    "h1-balance_simple-v0": 800.0,

    "h1-balance_hard-v0": 800.0,

    "h1-stair-v0": 700.0,

    "h1-slide-v0": 700.0,

    "h1-pole-v0": 700.0,

    "h1hand-walk-v0": 700.0,

    "h1hand-stand-v0": 800.0,

    "h1hand-run-v0": 700.0,

    "h1hand-reach-v0": 12000.0,

    "h1hand-hurdle-v0": 700.0,

    "h1hand-crawl-v0": 800.0,

    "h1hand-maze-v0": 1200.0,

    "h1hand-sit_simple-v0": 750.0,

    "h1hand-sit_hard-v0": 750.0,

    "h1hand-balance_simple-v0": 800.0,

    "h1hand-balance_hard-v0": 800.0,

    "h1hand-stair-v0": 700.0,

    "h1hand-slide-v0": 700.0,

    "h1hand-pole-v0": 700.0,

    "h1hand-push-v0": 700.0,

    "h1hand-cabinet-v0": 2500.0,

    "h1hand-highbar-v0": 750.0,

    "h1hand-door-v0": 600.0,

    "h1hand-truck-v0": 3000.0,

    "h1hand-cube-v0": 370.0,

    "h1hand-bookshelf_simple-v0": 2000.0,

    "h1hand-bookshelf_hard-v0": 2000.0,

    "h1hand-basketball-v0": 1200.0,

    "h1hand-window-v0": 650.0,

    "h1hand-spoon-v0": 650.0,

    "h1hand-kitchen-v0": 4.0,

    "h1hand-package-v0": 1500.0,

    "h1hand-powerlift-v0": 800.0,

    "h1hand-room-v0": 400.0,

    "h1hand-insert_small-v0": 350.0,

    "h1hand-insert_normal-v0": 350.0,

}

def normalize(res, name):

    return (res - HB_RANDOM_SCORE[name]) / (HB_SUCCESS_SCORE[name] - HB_RANDOM_SCORE[name])