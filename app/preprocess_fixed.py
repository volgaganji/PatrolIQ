import pandas as pd
import sys
df=pd.read_csv("data/sample_500k.csv", dtype=str, low_memory=False)
cols=list(df.columns)
print("columns_detected", cols)
lat_col=None
lon_col=None
date_col=None
for c in cols:
    cl=c.lower()
    if date_col is None and "date" in cl:
        date_col=c
    if lat_col is None and ("latitude" in cl or cl=="lat" or cl.endswith("_lat") or "lat " in cl or cl.startswith("lat")):
        lat_col=c
    if lon_col is None and ("longitude" in cl or cl=="lon" or cl.endswith("_lon") or "lon " in cl or cl.startswith("lon")):
        lon_col=c
if lat_col is None:
    for c in cols:
        if "y_coord" in c.lower() or "y_coordinate" in c.lower() or c.lower().endswith("_y"):
            lat_col=c
if lon_col is None:
    for c in cols:
        if "x_coord" in c.lower() or "x_coordinate" in c.lower() or c.lower().endswith("_x"):
            lon_col=c
if date_col is None or lat_col is None or lon_col is None:
    print("missing_required_columns", {"date_col":date_col,"lat_col":lat_col,"lon_col":lon_col})
    sys.exit(1)
df["date_temp"]=pd.to_datetime(df[date_col], errors="coerce")
df=df.dropna(subset=[lat_col,lon_col,"date_temp"])
df["date"]=df["date_temp"]
df=df.drop(columns=["date_temp"])
df["latitude"]=pd.to_numeric(df[lat_col], errors="coerce")
df["longitude"]=pd.to_numeric(df[lon_col], errors="coerce")
df=df.dropna(subset=["latitude","longitude","date"])
df["year"]=df["date"].dt.year
df["month"]=df["date"].dt.month
df["day"]=df["date"].dt.day
df["hour"]=df["date"].dt.hour
df["day_of_week"]=df["date"].dt.day_name()
df["is_weekend"]=df["day_of_week"].isin(["Saturday","Sunday"]).astype(int)
cols_to_keep=["id","case_number","iucr","fbi_code","primary_type","description","location_description","date","year","month","day","hour","day_of_week","is_weekend","block","latitude","longitude","x_coordinate","y_coordinate","beat","district","ward","community_area","arrest","domestic"]
final_cols=[c for c in cols_to_keep if c in df.columns] 
if "latitude" not in final_cols:
    final_cols.append("latitude")
if "longitude" not in final_cols:
    final_cols.append("longitude")
if "date" not in final_cols:
    final_cols.append("date")
df_out=df[final_cols]
df_out.to_parquet("data/preprocessed.parquet", index=False)
print("rows", len(df_out))
print("used_columns", {"date_col":date_col,"lat_col":lat_col,"lon_col":lon_col})
