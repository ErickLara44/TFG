import xarray as xr
import torch
import matplotlib.pyplot as plt

ds=xr.open_dataset("/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc")
fwi=ds["FWI"]

fwi_1 = ds.sel(time="2024-03-29")["FWI"].where(ds["is_spain"] == 1)
fwi_2 = ds.sel(time="2024-08-10")["FWI"].where(ds["is_spain"] == 1)
fwi_3 = ds.sel(time="2024-12-20")["FWI"].where(ds["is_spain"] == 1)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

p1 = fwi_1.plot(
    ax=axes[0],
    cmap="RdYlGn_r",
    vmin=0, vmax=50,
    add_colorbar=False
)
axes[0].axis("off")

p2 = fwi_2.plot(
    ax=axes[1],
    cmap="RdYlGn_r",
    vmin=0, vmax=50,
    add_colorbar=False
)
axes[1].axis("off")

p3 = fwi_3.plot(
    ax=axes[2],
    cmap="RdYlGn_r",
    vmin=0, vmax=50,
    add_colorbar=False
)
axes[2].axis("off")

axes[0].set_aspect("equal")
axes[1].set_aspect("equal")
axes[2].set_aspect("equal")

cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  
cbar = fig.colorbar(p3, cax=cbar_ax)
cbar.set_label('FWI')

plt.show()