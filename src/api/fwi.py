"""
Canadian Forest Fire Weather Index (FWI) System
================================================
Implements the full FWI System following the official Canadian Forest Service formulas:
  Van Wagner, C.E. (1987). Development and Structure of the Canadian Forest Fire
  Weather Index System. Can. For. Serv., Forestry Tech. Rep. 35.

Input variables (all must match the units used in the Datacube):
  - temp    : Temperature at noon (°C)
  - rh      : Relative Humidity at noon (%)
  - wind    : Wind speed at noon (km/h)  ← NOTE: FWI standard uses km/h!
  - rain    : 24-hour precipitation (mm)
  - month   : Month (1-12) for DC/DMC day-length adjustment
  - prev_ffmc: Previous day Fine Fuel Moisture Code (default 85.0)
  - prev_dmc : Previous day Duff Moisture Code (default 6.0)
  - prev_dc  : Previous day Drought Code (default 15.0)

Returns a dict with all sub-indices and the final FWI value.

Usage:
    from src.api.fwi import compute_fwi
    result = compute_fwi(temp=30, rh=25, wind=20, rain=0.0, month=8)
    print(result["FWI"])  # e.g. 42.3
"""

import math


# Day-length adjustment factors for DMC (by month)
_DMC_DAY_LENGTH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]

# Drying day-length factors for DC (by month) - per CFS table
_DC_DAY_LENGTH = [
    [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],  # 
]

# Latitude-based day-length lookup for DC (simplified European latitudes 46N+)
def _dc_lf(month: int) -> float:
    """DC day-length factor for mid-latitude Europe (~40-46°N)."""
    table = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    return table[month - 1]


def _ffmc(temp: float, rh: float, wind: float, rain: float, prev_ffmc: float) -> float:
    """Fine Fuel Moisture Code (FFMC)."""
    mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)  # Previous moisture content

    # Rain effect on FFMC
    if rain > 0.5:
        rf = rain - 0.5
        if mo <= 150.0:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        else:
            mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
            if mo > 250.0:
                mo = 250.0

    # Equilibrium moisture contents
    ed = 0.942 * (rh ** 0.679) + 11.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    ew = 0.618 * (rh ** 0.753) + 10.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))

    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * (wind ** 0.5) * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * (10.0 ** (-kd))
    elif mo < ew:
        kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * (wind ** 0.5) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
        kw = kl * 0.581 * math.exp(0.0365 * temp)
        m = ew - (ew - mo) * (10.0 ** (-kw))
    else:
        m = mo

    m = max(0.0, min(250.0, m))
    return 59.5 * (250.0 - m) / (147.2 + m)


def _dmc(temp: float, rh: float, rain: float, month: int, prev_dmc: float) -> float:
    """Duff Moisture Code (DMC)."""
    re = 0.92 * rain - 1.27 if rain > 1.5 else 0.0

    if re > 0.0:
        mo = 20.0 + math.exp(5.6348 - prev_dmc / 43.43)
        b = (
            100.0 / (0.5 + 0.3 * prev_dmc) if prev_dmc <= 33.0
            else (14.0 - 1.3 * math.log(prev_dmc) if prev_dmc <= 65.0
                  else 6.2 * math.log(prev_dmc) - 17.2)
        )
        mr = mo + (1000.0 * re) / (48.77 + b * re)
        pr = max(0.0, 244.72 - 43.43 * math.log(mr - 20.0))
    else:
        pr = prev_dmc

    if temp < -1.1:
        return pr

    k = 1.894 * (temp + 1.1) * (100.0 - rh) * _DMC_DAY_LENGTH[month - 1] * 1e-6
    return pr + 100.0 * k


def _dc(temp: float, rain: float, month: int, prev_dc: float) -> float:
    """Drought Code (DC)."""
    rd = 0.83 * rain - 1.27 if rain > 2.8 else 0.0

    if rd > 0.0:
        qo = 800.0 * math.exp(-prev_dc / 400.0)
        qr = qo + 3.937 * rd
        dr = max(0.0, 400.0 * math.log(800.0 / qr))
    else:
        dr = prev_dc

    if temp < -2.8:
        return dr

    v = 0.36 * (temp + 2.8) + _dc_lf(month)
    v = max(0.0, v)
    return dr + 0.5 * v


def _isi(wind: float, ffmc: float) -> float:
    """Initial Spread Index (ISI)."""
    fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fw = math.exp(0.05039 * wind)
    ff = 91.9 * math.exp(-0.1386 * fm) * (1.0 + (fm ** 5.31) / (4.93e7))
    return 0.208 * fw * ff


def _bui(dmc: float, dc: float) -> float:
    """Buildup Index (BUI)."""
    if dmc <= 0.4 * dc:
        u = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        u = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(0.0, u)


def _fwi(isi: float, bui: float) -> float:
    """Fire Weather Index (FWI)."""
    if bui <= 80.0:
        bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
    else:
        bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui)))
    return math.exp(2.72 * (0.434 * math.log(bb)) ** 0.647) if bb > 1.0 else bb


def compute_fwi(
    temp: float,
    rh: float,
    wind_kmh: float,
    rain: float,
    month: int,
    prev_ffmc: float = 85.0,
    prev_dmc: float = 6.0,
    prev_dc: float = 15.0,
) -> dict:
    """
    Compute the full Canadian FWI System given today's weather.

    Args:
        temp     : Noon temperature (°C)
        rh       : Noon relative humidity (%)
        wind_kmh : Noon wind speed (km/h)  ← FWI uses km/h, not m/s!
        rain     : 24-hour precipitation (mm)
        month    : Calendar month (1-12)
        prev_ffmc: Previous FFMC (default 85.0 = standard startup)
        prev_dmc : Previous DMC  (default 6.0  = standard startup)
        prev_dc  : Previous DC   (default 15.0 = standard startup)

    Returns:
        dict with keys: FFMC, DMC, DC, ISI, BUI, FWI
    """
    # Clamp inputs to physically valid ranges
    rh = max(0.0, min(100.0, rh))
    wind_kmh = max(0.0, wind_kmh)
    rain = max(0.0, rain)

    ffmc = _ffmc(temp, rh, wind_kmh, rain, prev_ffmc)
    dmc  = _dmc(temp, rh, rain, month, prev_dmc)
    dc   = _dc(temp, rain, month, prev_dc)
    isi  = _isi(wind_kmh, ffmc)
    bui  = _bui(dmc, dc)
    fwi  = _fwi(isi, bui)

    return {
        "FFMC": round(ffmc, 2),
        "DMC":  round(dmc, 2),
        "DC":   round(dc, 2),
        "ISI":  round(isi, 2),
        "BUI":  round(bui, 2),
        "FWI":  round(fwi, 2),
    }
