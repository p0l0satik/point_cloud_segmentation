import pandas as pd
# from matplotlib.colors import rgb2hex
# import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import ipyvolume as ipv
import numpy as np
x, y, z = np.random.random((3, 10000))
ipv.quickscatter(x, y, z, size=1, marker="sphere")