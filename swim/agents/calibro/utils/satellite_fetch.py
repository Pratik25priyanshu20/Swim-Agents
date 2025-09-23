# swim/agents/calibro/satellite_fetch.py

import ee
import math

# Morphology kernel sizes (used for water masking)
ERODE_PIXELS = 2
BUFFER_PIXELS = 3
MIN_PATCH_PX = 8

# NDCI to Chl-a Coefficients
NDCI_COEFFS = dict(a0=-0.40, a1=1.10, a2=0.00)

# Dogliotti turbidity algorithm coefficients
DOG_RED = dict(lambda_nm=665, A=230.0, C=0.170)
DOG_NIR = dict(lambda_nm=865, A=1300.0, C=0.212)
DOG_SWITCH_RRS = 0.03


# ----------------------------------
# Geometry Helper
# ----------------------------------

def lake_to_aoi(lake: dict) -> ee.Geometry:
    return ee.Geometry.Point([lake["lon"], lake["lat"]]).buffer(lake["buffer_km"] * 1000.0)


# ----------------------------------
# Masking & Preprocessing Functions
# ----------------------------------

def mask_s2_clouds_shadows(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    qa60 = image.select("QA60")
    scl_good = (scl.neq(1)
                .And(scl.neq(3))
                .And(scl.neq(7)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                .And(scl.neq(11)))
    cloud_bit, cirrus_bit = 1 << 10, 1 << 11
    qa_clear = qa60.bitwiseAnd(cloud_bit).eq(0).And(qa60.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(scl_good.And(qa_clear))


def add_rrs_from_sr(image: ee.Image) -> ee.Image:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    rrs_bands = [image.select(b).divide(math.pi).rename(f"Rrs_{b}") for b in bands]
    return image.addBands(ee.Image.cat(rrs_bands))


def add_indices(image: ee.Image) -> ee.Image:
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    rrs705 = image.select("Rrs_B5")
    rrs665 = image.select("Rrs_B4")
    ndci = rrs705.subtract(rrs665).divide(rrs705.add(rrs665)).rename("NDCI")
    tss_rel = image.select("Rrs_B6").rename("TSS_rel")
    return image.addBands([ndwi, mndwi, ndci, tss_rel])


def build_water_mask(image: ee.Image) -> ee.Image:
    ndwi, mndwi = image.select("NDWI"), image.select("MNDWI")
    scl = image.select("SCL")
    choice = ndwi.where(mndwi.gt(ndwi), mndwi)
    water0 = choice.gt(0).Or(scl.eq(6))
    k_erode = ee.Kernel.square(ERODE_PIXELS)
    k_buffer = ee.Kernel.square(BUFFER_PIXELS)
    eroded = water0.reduceNeighborhood(ee.Reducer.min(), k_erode).gt(0)
    buffered = eroded.reduceNeighborhood(ee.Reducer.min(), k_buffer).gt(0)
    ccount = buffered.selfMask().connectedPixelCount(maxSize=256, eightConnected=True)
    cleaned = buffered.updateMask(ccount.gte(MIN_PATCH_PX))
    return cleaned.rename("WATER_MASK")



# ----------------------------------
# Water Quality Products
# ----------------------------------

def chl_from_ndci(image: ee.Image) -> ee.Image:
    a0, a1, a2 = NDCI_COEFFS["a0"], NDCI_COEFFS["a1"], NDCI_COEFFS["a2"]
    ndci = image.select("NDCI")
    log10_chl = ndci.multiply(a1).add(a0).add(ndci.pow(2).multiply(a2)).rename("log10_Chl")
    chl = ee.Image(10).pow(log10_chl).rename("Chl_mg_m3")
    return image.addBands([log10_chl, chl])


def dogliotti_turbidity(image: ee.Image) -> ee.Image:
    rrs_red = image.select("Rrs_B4")   # ~665 nm
    rrs_nir = image.select("Rrs_B8A")  # ~865 nm

    def dog(rrs: ee.Image, A: float, C: float) -> ee.Image:
        return rrs.multiply(A).divide(ee.Image(1).subtract(rrs.divide(C)))

    t_red = dog(rrs_red, DOG_RED["A"], DOG_RED["C"]).rename("Turb_red")
    t_nir = dog(rrs_nir, DOG_NIR["A"], DOG_NIR["C"]).rename("Turb_nir")
    use_nir = rrs_red.gte(DOG_SWITCH_RRS)
    turb = t_red.where(use_nir, t_nir).rename("Turbidity_FNU")
    return image.addBands([t_red, t_nir, turb])





# ----------------------------------
# Quality Filters
# ----------------------------------

def percentiles_25_50_75(ic: ee.ImageCollection, band: str):
    p = ic.select(band).reduce(ee.Reducer.percentile([25, 50, 75]))
    q1  = p.select(f"{band}_p25")
    med = p.select(f"{band}_p50").rename(f"{band}_median")
    q3  = p.select(f"{band}_p75")
    iqr = q3.subtract(q1).rename(f"{band}_IQR")
    return q1, med, q3, iqr


def spatial_outlier_mask(img: ee.Image, band: str) -> ee.Image:
    x = img.select(band)
    kernel = ee.Kernel.square(radius=1)

    mean = x.reduceNeighborhood(ee.Reducer.mean(), kernel)
    std = x.reduceNeighborhood(ee.Reducer.stdDev(), kernel)
    std = std.where(std.eq(0), 1e-6)
    z_ok = x.subtract(mean).divide(std).abs().lte(3.0)

    med = x.reduceNeighborhood(ee.Reducer.median(), kernel)
    mad = x.subtract(med).abs().reduceNeighborhood(ee.Reducer.median(), kernel)
    mad = mad.where(mad.eq(0), 1e-6)
    r_ok = x.subtract(med).divide(mad.multiply(1.4826)).abs().lte(3.0)

    return z_ok.And(r_ok).rename(f"{band}_SPATIAL_QA")