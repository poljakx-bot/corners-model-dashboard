import streamlit as st
import pandas as pd
import os
import glob
import sys
import hashlib

sys.path.append(os.getcwd())
import bookmaker_kornerv10 as model

# ====================== PASSWORD ZAŠTITA ======================
PASSWORD = "tvoja_lozinka_2026"   # ← PROMENI OVO U NEŠTO JAKO!

def check_password():
    """Vrati True ako je password tačan"""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.text_input(
            "Unesi lozinku za pristup",
            type="password",
            key="password",
            on_change=check_password_callback
        )
        return False
    return True

def check_password_callback():
    if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == hashlib.sha256(PASSWORD.encode()).hexdigest():
        st.session_state.password_correct = True
        st.rerun()
    else:
        st.error("❌ Pogrešna lozinka")

# ====================== CONFIG ======================
if not check_password():
    st.stop()   # zaustavlja dalje učitavanje

st.set_page_config(page_title="Corners Model Dashboard", layout="wide")
st.title("🏟️ Corners Model Dashboard")
st.markdown("**Premier League 2025/26 | Privatni dashboard**")

DATA_DIR = "data"

# ====================== LOAD DATA ======================
@st.cache_data
def load_all_gw():
    files = glob.glob(os.path.join(DATA_DIR, "bookmaker_odds_corners_gw*.csv"))
    dfs = {}
    for f in sorted(files, key=lambda x: int(x.split("gw")[1].split("_")[0])):
        gw = int(f.split("gw")[1].split("_")[0])
        df = pd.read_csv(f)
        dfs[gw] = df
    return dfs

all_gw = load_all_gw()
gw_list = sorted(all_gw.keys())

# ====================== DINAMIČKI μ LEAGUE (uvek svež) ======================
def get_current_mu_and_gw():
    """Računa μ i GW iz NAJNOVIJEG fajla (bez keša)"""
    files = glob.glob(os.path.join(DATA_DIR, "bookmaker_odds_corners_gw*.csv"))
    if not files:
        return 9.84, "N/A"
    
    latest_file = max(files, key=os.path.getmtime)
    latest_gw = int(latest_file.split("gw")[1].split("_")[0])
    
    df_latest = pd.read_csv(latest_file)
    mu = float(df_latest['mu_league'].iloc[0]) if 'mu_league' in df_latest.columns else 9.84
    
    return mu, latest_gw

current_mu, current_gw = get_current_mu_and_gw()

# ====================== LOAD FITTED MODEL ======================
@st.cache_data
def load_fitted_model():
    teams_path = os.path.join(DATA_DIR, "model_teams_table_gw26.csv")
    if os.path.exists(teams_path):
        df_teams = pd.read_csv(teams_path)
        fitted = model.FittedModel(
            teams=df_teams['team'].tolist(),
            beta0=0.0,
            home_adv=0.0,
            alpha=0.03,
            attack={row['team']: row['attack'] for _, row in df_teams.iterrows()},
            defense={row['team']: row['defense'] for _, row in df_teams.iterrows()},
            matches_w={row['team']: row['matches_w'] for _, row in df_teams.iterrows()},
            team_tempo_factor={row['team']: row['tempo'] for _, row in df_teams.iterrows()}
        )
        sanity_path = os.path.join(DATA_DIR, "model_sanity_table_gw26.csv")
        if os.path.exists(sanity_path):
            sanity = pd.read_csv(sanity_path)
            fitted.beta0 = float(sanity[sanity['metric'] == 'beta0']['value'].iloc[0])
            fitted.home_adv = float(sanity[sanity['metric'] == 'home_adv']['value'].iloc[0])
        return fitted
    return None

fitted_model = load_fitted_model()

# ====================== SIDEBAR ======================
st.sidebar.header("🔧 Globalni parametri")

# VELIKI BOLD METRIK SA GW
st.sidebar.metric(
    "**Aktuelni μ League**", 
    f"**{current_mu:.3f}** (GW{current_gw})",
    delta=None
)

margin = st.sidebar.slider("Margin", 0.010, 0.120, 0.080, step=0.005)
alpha_squeeze = st.sidebar.slider("Alpha Squeeze - priblizi srednjoj vrednosti", 0.50, 0.90, 0.68, step=0.01)
over_price_boost = st.sidebar.slider("Over Price Boost", 0.00, 0.03, 0.018, step=0.001)
use_tempo = st.sidebar.checkbox("Use Tempo Layer", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("μ League + GW se ažurira posle svakog Recompute")

# ====================== TABS ======================
tab1, tab2 = st.tabs(["📊 Existing GW (Recompute)", "🆕 Custom Matches"])

# ====================== TAB 1: EXISTING GW ======================
with tab1:
    st.subheader("Odaberi Gameweek")
    selected_gw = st.selectbox("Gameweek", gw_list, index=len(gw_list)-1)
    
    if st.button("🔄 Recompute GW", type="primary"):
        with st.spinner(f"Računam GW {selected_gw}..."):
            model.USE_TEMPO_LAYER = use_tempo
            model.ALPHA_SQUEEZE_DEFAULT = alpha_squeeze
            model.main(
                margin=margin,
                over_price_boost=over_price_boost,
                alpha_squeeze=alpha_squeeze
            )
        st.success(f"✅ GW {selected_gw} recomputed! μ League je sada izračunat iz GW{selected_gw}.")
        st.rerun()  # OSVEŽAVA I μ METRIK
    
    df = all_gw[selected_gw]
    st.subheader(f"GW {selected_gw} – {len(df)} mečeva")
    
    # Kolone
    base_cols = ["home_team", "away_team", "line", "p_over", "bookmaker_odds_over", 
                 "bookmaker_odds_under", "mu_match"]
    extra_cols = ["p_over_absdiff_5050", "mu_home", "mu_away", "alpha_used"]
    cols_to_show = [col for col in base_cols + extra_cols if col in df.columns]
    
    format_dict = {
        "p_over": "{:.1%}",
        "bookmaker_odds_over": "{:.2f}",
        "bookmaker_odds_under": "{:.2f}",
        "mu_match": "{:.2f}",
    }
    if "p_over_absdiff_5050" in df.columns:
        format_dict["p_over_absdiff_5050"] = "{:.3f}"
    
    st.dataframe(
        df[cols_to_show].style.format(format_dict),
        use_container_width=True,
        height=650
    )
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, f"corners_gw{selected_gw}.csv", "text/csv")

# ====================== TAB 2: CUSTOM MATCHES ======================
with tab2:
    st.subheader("🆕 Custom GW")
    custom_input = st.text_area(
        "Unesi mečeve (Home vs Away po liniji)",
        height=180,
        placeholder="Liverpool vs Sunderland\nTottenham vs Newcastle"
    )
    
    if st.button("🚀 Izračunaj custom kvote", type="primary"):
        if not custom_input.strip():
            st.error("Unesi mečeve!")
        elif fitted_model is None:
            st.error("Nema modela!")
        else:
            with st.spinner(f"Računam sa μ={current_mu:.3f} (GW{current_gw})..."):
                model.USE_TEMPO_LAYER = use_tempo
                model.ALPHA_SQUEEZE_DEFAULT = alpha_squeeze
                lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
                
                custom_rows = []
                for line in custom_input.strip().split("\n"):
                    if "vs" in line.lower():
                        parts = [p.strip() for p in line.split("vs", 1)]
                        if len(parts) == 2:
                            home, away = [model.normalize_team_name(p) for p in parts]
                            try:
                                df_match = model.compute_fixture_odds(
                                    fitted_model, home, away, lines, margin, current_mu, 
                                    over_price_boost, alpha_squeeze
                                )
                                df_match = df_match.assign(gameweek=99, home_team=home, away_team=away)
                                custom_rows.append(df_match)
                            except Exception as e:
                                st.error(f"Greška za {home} vs {away}: {str(e)[:80]}")
                
                if custom_rows:
                    custom_df = pd.concat(custom_rows, ignore_index=True)
                    st.success(f"✅ {len(custom_df)} linija sa μ={current_mu:.3f} (GW{current_gw})!")
                    st.dataframe(custom_df[["home_team", "away_team", "line", "p_over", 
                                          "bookmaker_odds_over", "bookmaker_odds_under", "mu_match"]].style.format({
                        "p_over": "{:.1%}", "bookmaker_odds_over": "{:.2f}", 
                        "bookmaker_odds_under": "{:.2f}", "mu_match": "{:.2f}"
                    }), height=500)
                    
                    csv = custom_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Custom", csv, "custom_corners.csv", "text/csv")

# Sidebar info
st.sidebar.metric("Model", "GW26")