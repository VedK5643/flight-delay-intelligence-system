import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Flight Delay Intelligence System",
    page_icon="✈",
    layout="wide"
)

CARRIER_NAMES = {
    "WN":"Southwest","DL":"Delta","AA":"American","UA":"United",
    "OO":"SkyWest","EV":"ExpressJet","B6":"JetBlue","MQ":"Envoy Air",
    "AS":"Alaska","US":"US Airways","NK":"Spirit","F9":"Frontier",
    "HA":"Hawaiian","VX":"Virgin America","YX":"Republic","OH":"PSA",
    "9E":"Endeavor","YV":"Mesa","FL":"AirTran","G4":"Allegiant",
}
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
DAY_NAMES   = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    # Works whether you run from project root or dashboard/
    for path in [
        "data/processed/flights_processed.parquet",
        "../data/processed/flights_processed.parquet",
    ]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "ARR_DEL15" not in df.columns:
                df["ARR_DEL15"] = (df["ARR_DELAY"] >= 15).astype(int)
            df["CARRIER_NAME"] = df["OP_CARRIER"].map(CARRIER_NAMES).fillna(df["OP_CARRIER"])
            return df
    st.error("Could not find flights_processed.csv. Run from project root: streamlit run app.py")
    st.stop()

@st.cache_resource
def load_model():
    for path in [
        "models/xgb_model.pkl",
        "../models/xgb_model.pkl",
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

df = load_data()

@st.cache_data
def get_encodings(_df):
    carrier_enc = _df.groupby("OP_CARRIER")["ARR_DEL15"].mean().to_dict()
    origin_enc  = _df.groupby("ORIGIN")["ARR_DEL15"].mean().to_dict()
    dest_enc    = _df.groupby("DEST")["ARR_DEL15"].mean().to_dict()
    return carrier_enc, origin_enc, dest_enc

@st.cache_data
def get_route_distances(_df):
    return _df.groupby(['ORIGIN','DEST'])['DISTANCE'].first().to_dict()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("✈ Flight Delay Intelligence")
st.sidebar.caption("BTS On-Time Performance · 2014–2018")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🔮 Delay Predictor",
    "✈ Airline Analytics",
    "🗺 Airport Intelligence",
    "⚠ Delay Cause Analysis",
    "🔬 SHAP Explainability",
])

st.sidebar.markdown("---")
st.sidebar.caption(f"Flights: {len(df):,}")
st.sidebar.caption("Model: XGBoost · ROC-AUC 0.671")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Flight Delay Intelligence System")
    st.caption("Predictive analytics platform built on 2.9M domestic US flights · 2014–2018")
    st.markdown("---")

    # KPIs
    total     = len(df)
    delay_rt  = df["ARR_DEL15"].mean() * 100
    carriers  = df["OP_CARRIER"].nunique()
    airports  = df["ORIGIN"].nunique()
    avg_delay = df[df["ARR_DELAY"] > 0]["ARR_DELAY"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Flights Analysed",   f"{total/1e6:.2f}M")
    c2.metric("Overall Delay Rate", f"{delay_rt:.1f}%")
    c3.metric("Airlines",           carriers)
    c4.metric("Airports",           airports)
    c5.metric("Avg Delay (when late)", f"{avg_delay:.0f} min")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Delay Rate by Month")
        monthly = df.groupby("MONTH")["ARR_DEL15"].mean().mul(100).round(2).sort_index()
        monthly.index = [MONTH_NAMES[m] for m in monthly.index]
        st.bar_chart(monthly)

    with col_r:
        st.subheader("Delay Rate by Departure Hour")
        hourly = df.groupby("CRS_DEP_TIME")["ARR_DEL15"].mean().mul(100).round(2)
        st.line_chart(hourly)

    st.markdown("---")
    st.subheader("Delay Rate by Day of Week")
    daily = df.groupby("DAY_OF_WEEK")["ARR_DEL15"].mean().mul(100).round(2)
    daily.index = [DAY_NAMES[d] for d in daily.index]
    st.bar_chart(daily)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DELAY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Delay Predictor":
    st.title("Delay Predictor")
    st.caption("Enter flight details to get a delay probability from the XGBoost model.")
    st.markdown("---")

    carrier_enc, origin_enc, dest_enc = get_encodings(df)
    route_dist_map = get_route_distances(df)
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Flight Details")

        carriers_list   = sorted(df["OP_CARRIER"].unique())
        carrier_display = [f"{c} — {CARRIER_NAMES.get(c, c)}" for c in carriers_list]
        carrier_sel     = st.selectbox("Airline", carrier_display)
        carrier_code    = carrier_sel.split(" — ")[0]

        airports_list = sorted(df["ORIGIN"].unique())
        origin = st.selectbox("Origin Airport", airports_list,
                              index=airports_list.index("ATL") if "ATL" in airports_list else 0)
        dest   = st.selectbox("Destination Airport", airports_list,
                              index=airports_list.index("LAX") if "LAX" in airports_list else 1)

        c1, c2 = st.columns(2)
        dep_hour = c1.slider("Departure Hour", 0, 23, 8)
        arr_hour = c2.slider("Arrival Hour",   0, 23, 11)

        c3, c4 = st.columns(2)
        month = c3.selectbox("Month", list(MONTH_NAMES.keys()), format_func=lambda x: MONTH_NAMES[x], index=5)
        dow   = c4.selectbox("Day of Week", list(DAY_NAMES.keys()), format_func=lambda x: DAY_NAMES[x])

        
        distance = int(route_dist_map.get((origin, dest), df['DISTANCE'].mean()))

        st.info(f"Route distance: {distance} miles")
        predict = st.button("Predict", width='stretch', type="primary")

    with col_result:
        st.subheader("Result")

        if predict:
            model = load_model()
            if model is None:
                st.error("Model not found. Save xgb_model.pkl to models/ folder.")
            else:
                features = np.array([[
                    dep_hour, arr_hour, distance, month, dow,
                    carrier_enc.get(carrier_code, 0.18),
                    origin_enc.get(origin, 0.18),
                    dest_enc.get(dest, 0.18),
                ]])
                prob    = model.predict_proba(features)[0][1]

                if prob >= 0.5:
                    risk_label = "🔴 HIGH RISK"
                    st.error(f"⚠ Likely Delayed")
                elif prob >= 0.35:
                    risk_label = "🟡 MEDIUM RISK"
                    st.warning(f"⚠ Borderline")
                else:
                    risk_label = "🟢 LOW RISK"
                    st.success(f"✅ Likely On Time")

                st.metric("Delay Probability", f"{prob*100:.1f}%", delta=risk_label, delta_color="off")
                st.progress(float(prob))

                st.markdown("---")
                st.markdown(f"""
**Route:** {origin} → {dest}  
**Carrier:** {CARRIER_NAMES.get(carrier_code, carrier_code)}  
**Departure:** {dep_hour:02d}:00  
**Month:** {MONTH_NAMES[month]} · **Day:** {DAY_NAMES[dow]}  
**Distance:** {distance} miles
""")
        else:
            st.info("Fill in flight details and click Predict.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AIRLINE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✈ Airline Analytics":
    st.title("Airline Analytics")
    st.caption("Performance comparison across all 20 carriers.")
    st.markdown("---")

    airline_stats = (
        df.groupby(["OP_CARRIER","CARRIER_NAME"])
        .agg(
            Flights    =("ARR_DEL15","count"),
            Delay_Rate =("ARR_DEL15","mean"),
            Avg_Delay  =("ARR_DELAY", lambda x: x[x > 0].mean()),
        )
        .reset_index()
    )
    airline_stats["Delay Rate %"]   = (airline_stats["Delay_Rate"] * 100).round(1)
    airline_stats["Reliability %"]  = (100 - airline_stats["Delay Rate %"]).round(1)
    airline_stats["Avg Delay (min)"]= airline_stats["Avg_Delay"].round(1)
    airline_stats = airline_stats.sort_values("Delay Rate %")

    col_b, col_w = st.columns(2)

    with col_b:
        st.subheader("🏆 Most Reliable")
        for i, (_, row) in enumerate(airline_stats.head(3).iterrows()):
            medal = ["🥇","🥈","🥉"][i]
            st.success(f"{medal} **{row['CARRIER_NAME']}** — {row['Delay Rate %']}% delay rate")

    with col_w:
        st.subheader("⚠ Most Delayed")
        for _, row in airline_stats.tail(3).iloc[::-1].iterrows():
            st.error(f"⚠ **{row['CARRIER_NAME']}** — {row['Delay Rate %']}% delay rate")

    st.markdown("---")
    st.subheader("Delay Rate by Airline")
    chart_data = airline_stats.set_index("CARRIER_NAME")["Delay Rate %"]
    st.bar_chart(chart_data)

    st.markdown("---")
    st.subheader("Full Rankings")
    display = airline_stats[["CARRIER_NAME","OP_CARRIER","Flights","Delay Rate %","Reliability %","Avg Delay (min)"]].copy()
    display.columns = ["Airline","Code","Flights","Delay Rate %","Reliability %","Avg Delay (min)"]
    display["Flights"] = display["Flights"].apply(lambda x: f"{x:,}")
    display = display.reset_index(drop=True)
    display.index += 1
    st.dataframe(display, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AIRPORT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 Airport Intelligence":
    st.title("Airport Intelligence")
    st.caption("Delay hotspots across 367 US airports.")
    st.markdown("---")

    n = st.slider("Number of airports", 10, 50, 20, 5)

    airport_stats = (
        df.groupby("ORIGIN")
        .agg(Flights=("ARR_DEL15","count"), Delay_Rate=("ARR_DEL15","mean"))
        .reset_index()
    )
    airport_stats["Delay Rate %"] = (airport_stats["Delay_Rate"] * 100).round(2)
    top_n = airport_stats.nlargest(n, "Flights")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Delay Rate (Top Busiest)")
        chart = top_n.set_index("ORIGIN")["Delay Rate %"].sort_values()
        st.bar_chart(chart)

    with col_r:
        st.subheader("Flight Volume")
        chart2 = top_n.set_index("ORIGIN")["Flights"].sort_values()
        st.bar_chart(chart2)

    st.markdown("---")
    st.subheader("Riskiest & Safest Routes")

    route_stats = (
        df.groupby(["ORIGIN","DEST"])
        .agg(Flights=("ARR_DEL15","count"), Delay_Rate=("ARR_DEL15","mean"))
        .reset_index()
    )
    route_stats = route_stats[route_stats["Flights"] >= 500]
    route_stats["Delay %"] = (route_stats["Delay_Rate"] * 100).round(1)
    route_stats["Route"]   = route_stats["ORIGIN"] + " → " + route_stats["DEST"]

    col_risk, col_safe = st.columns(2)

    with col_risk:
        st.markdown("**⚠ Riskiest Routes**")
        risky = route_stats.nlargest(10,"Delay %")[["Route","Flights","Delay %"]].reset_index(drop=True)
        risky.index += 1
        st.dataframe(risky, width='stretch')

    with col_safe:
        st.markdown("**✅ Safest Routes**")
        safe = route_stats.nsmallest(10,"Delay %")[["Route","Flights","Delay %"]].reset_index(drop=True)
        safe.index += 1
        st.dataframe(safe, width='stretch')

elif page == "🔬 SHAP Explainability":
    st.title("SHAP Explainability")
    st.caption("Why does the model predict a delay? SHAP values explain each feature's contribution.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Global Feature Importance")
        st.caption("Average SHAP impact across all predictions.")
        for path in ["reports/shap_importance.png", "../reports/shap_importance.png"]:
            if os.path.exists(path):
                st.image(path)
                break

    with col2:
        st.subheader("Feature Impact Direction")
        st.caption("Red = high feature value, Blue = low. Right = increases delay probability.")
        for path in ["reports/shap_beeswarm.png", "../reports/shap_beeswarm.png"]:
            if os.path.exists(path):
                st.image(path)
                break

    st.markdown("---")
    st.subheader("Key Insights")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**What drives delays most:**
- 🕐 **Departure Hour** — strongest predictor. 5AM = ~7% delay rate, 8PM = ~27%
- ✈ **Airline** — carrier historical delay rate is second most important
- 📅 **Month** — June/July worst, September best
        """)
    with c2:
        st.markdown("""
**What matters less:**
- 📍 **Distance** — least important. Long/short flights delay at similar rates
- 🗺 **Origin/Dest** — moderate importance, less than time and carrier

**Bottom line:** Book 5-6AM on Hawaiian or Delta in September for lowest delay risk.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DELAY CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠ Delay Cause Analysis":
    st.title("Delay Cause Analysis")
    st.caption("What actually causes flight delays.")
    st.markdown("---")

    causes       = ["CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"]
    cause_labels = ["Carrier","Weather","NAS / ATC","Security","Late Aircraft"]

    delayed_df = df[df["ARR_DEL15"] == 1]

    # Overall totals
    st.subheader("Total Delay Minutes by Cause")
    totals = delayed_df[causes].sum()
    totals.index = cause_labels
    st.bar_chart(totals)

    st.info("**Late Aircraft** is the #1 cause — a delayed incoming plane cascades into the next departure. This is why evening flights are consistently worse than morning flights.")

    st.markdown("---")

    # Per airline
    st.subheader("Cause Breakdown by Airline")
    carrier_sel = st.selectbox("Select airline", sorted(df["CARRIER_NAME"].unique()))
    carrier_data = delayed_df[delayed_df["CARRIER_NAME"] == carrier_sel][causes].sum()
    carrier_data.index = cause_labels
    st.bar_chart(carrier_data)

    st.markdown("---")

    # Seasonal
    st.subheader("Monthly Delay Pattern by Cause")
    monthly_causes = delayed_df.groupby("MONTH")[causes].mean().sort_index()
    monthly_causes.index = [MONTH_NAMES[m] for m in monthly_causes.index]
    monthly_causes.columns = cause_labels
    st.line_chart(monthly_causes)

    st.info("**June–August:** Weather delays spike — peak US thunderstorm season. **December–January:** NAS delays rise due to winter storms.")
