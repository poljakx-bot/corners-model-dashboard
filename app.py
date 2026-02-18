import streamlit as st

# IMPORTANT: set_page_config MUST be called before any other Streamlit commands
st.set_page_config(page_title="Corners Model Dashboard", layout="wide")

import pandas as pd
import os
import glob
import sys
import hashlib

# Allow importing local model module from repo root
sys.path.append(os.getcwd())

# Prefer v11; fall back to v10 if repo doesn't have v11 yet
MODEL_MODULE_CANDIDATES = [
    "bookmaker_kornerv11_logging",
    "bookmaker_kornerv10",
]

model = None
_model_import_errs = []
for _m in MODEL_MODULE_CANDIDATES:
    try:
        model = __import__(_m)
        MODEL_MODULE_NAME = _m
        break
    except Exception as e:
        _model_import_errs.append(f"{_m}: {e}")

if model is None:
    st.error("❌ Ne mogu da importujem model modul. Pokušao sam:\n\n- " + "\n- ".join(_model_import_errs))
    st.stop()


# ====================== PASSWORD ZAŠTITA ======================
# NOTE: promeni lozinku pre deploy-a (nikad ne drži pravu lozinku u repo-u)
PASSWORD = "tvoja_lozinka_2026"  # ← PROMENI OVO U NEŠTO JAKO!

def check_password():
    """Vrati True ako je password tačan."""
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


# ====================== START APP ======================
if not check_password():
    st.stop()

st.title("🏟️ Corners Model Dashboard")
st.markdown("**Premier League 2025/26 | Privatni dashboard**")

DATA_DIR = "data"


# ====================== HELPERS ======================
def _extract_gw_from_filename(path: str) -> int | None:
    base = os.path.basename(path)
    if "gw" not in base:
        return None
    try:
        return int(base.split("gw")[1].split("_")[0].split(".")[0])
    except Exception:
        return None

def _get_latest_file_per_gw(pattern: str) -> dict[int, str]:
    """Return map gw -> newest file path by mtime."""
    files = glob.glob(os.path.join(DATA_DIR, pattern))
    by_gw: dict[int, str] = {}
    for f in files:
        gw = _extract_gw_from_filename(f)
        if gw is None:
            continue
        if (gw not in by_gw) or (os.path.getmtime(f) > os.path.getmtime(by_gw[gw])):
            by_gw[gw] = f
    return by_gw

def _safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ====================== LOAD DATA ======================
@st.cache_data
def load_all_gw_latest():
    """Loads newest CSV per GW (handles multiple margin variants)."""
    by_gw = _get_latest_file_per_gw("bookmaker_odds_corners_gw*.csv")
    dfs: dict[int, pd.DataFrame] = {}
    for gw in sorted(by_gw.keys()):
        dfs[gw] = pd.read_csv(by_gw[gw])
    return dfs

def get_current_mu_and_gw():
    """Računa μ i GW iz NAJNOVIJEG fajla (bez keša)."""
    files = glob.glob(os.path.join(DATA_DIR, "bookmaker_odds_corners_gw*.csv"))
    if not files:
        return 9.84, "N/A"
    latest_file = max(files, key=os.path.getmtime)
    latest_gw = _extract_gw_from_filename(latest_file) or "N/A"
    df_latest = pd.read_csv(latest_file)
    mu_col = _safe_col(df_latest, ["mu_league", "mu_pl_league", "mu"])
    mu = float(df_latest[mu_col].iloc[0]) if mu_col else 9.84
    return mu, latest_gw

all_gw = load_all_gw_latest()
gw_list = sorted(all_gw.keys())
current_mu, current_gw = get_current_mu_and_gw()


# ====================== LOAD FITTED MODEL (optional for custom) ======================
@st.cache_data
def load_fitted_model():
    """
    Loads team params from a CSV if present (your precomputed table),
    so Custom Matches tab can work without re-fitting.
    """
    teams_path = os.path.join(DATA_DIR, "model_teams_table_gw26.csv")
    if not os.path.exists(teams_path):
        return None

    df_teams = pd.read_csv(teams_path)

    # Try to build a minimal fitted model object compatible with compute_fixture_odds()
    # If v11 has extra fields they should have defaults.
    fitted = model.FittedModel(
        teams=df_teams["team"].tolist(),
        beta0=0.0,
        home_adv=0.0,
        alpha=0.03,
        attack={row["team"]: float(row["attack"]) for _, row in df_teams.iterrows()},
        defense={row["team"]: float(row["defense"]) for _, row in df_teams.iterrows()},
        matches_w={row["team"]: float(row["matches_w"]) for _, row in df_teams.iterrows()},
        team_tempo_factor={row["team"]: float(row["tempo"]) for _, row in df_teams.iterrows()},
    )

    sanity_path = os.path.join(DATA_DIR, "model_sanity_table_gw26.csv")
    if os.path.exists(sanity_path):
        sanity = pd.read_csv(sanity_path)

        def _get_metric(name: str, default=None):
            try:
                return float(sanity.loc[sanity["metric"] == name, "value"].iloc[0])
            except Exception:
                return default

        fitted.beta0 = _get_metric("beta0", fitted.beta0)
        fitted.home_adv = _get_metric("home_adv", fitted.home_adv)

        # Optional (if present in sanity table)
        mu_l = _get_metric("mu_league", None)
        if mu_l is not None and hasattr(fitted, "mu_league"):
            fitted.mu_league = mu_l

    # If model expects mu_league but it wasn't in sanity table, align with current_mu
    if hasattr(fitted, "mu_league") and (getattr(fitted, "mu_league", None) in (None, 0)):
        try:
            fitted.mu_league = float(current_mu)
        except Exception:
            pass

    return fitted

fitted_model = load_fitted_model()


# ====================== SIDEBAR ======================
st.sidebar.header("🔧 Globalni parametri")

st.sidebar.metric(
    "Aktuelni μ League",
    f"{current_mu:.3f} (GW{current_gw})"
)

st.sidebar.caption(f"Model modul: **{MODEL_MODULE_NAME}**")

margin = st.sidebar.slider("Margin", 0.000, 0.120, 0.080, step=0.005)
alpha_squeeze = st.sidebar.slider("Suzi raspodelu", 0.50, 9.90, 0.68, step=0.01)
over_price_boost = st.sidebar.slider("Pomeri kvotu za Over", 0.00, 0.10, 0.01, step=0.005)
use_tempo = st.sidebar.checkbox("Use Tempo Layer", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("μ League + GW se ažurira posle svakog Recompute (na osnovu najnovijeg CSV-a).")


# ====================== TABS ======================
tab1, tab2 = st.tabs(["📊 Existing GW (Recompute)", "🆕 Custom Matches"])


# ====================== TAB 1: EXISTING GW ======================
with tab1:
    if not gw_list:
        st.warning("Nema dostupnih GW CSV fajlova u data folderu.")
    else:
        st.subheader("Odaberi Gameweek")
        selected_gw = st.selectbox("Gameweek", gw_list, index=len(gw_list) - 1)

        if st.button("🔄 Recompute CURRENT GW (iz active season fajla)", type="primary"):
            with st.spinner("Računam current GW iz active season fajla..."):
                # Pass toggles into model globals (as you already do)
                try:
                    model.USE_TEMPO_LAYER = use_tempo
                except Exception:
                    pass
                try:
                    model.ALPHA_SQUEEZE_DEFAULT = alpha_squeeze
                except Exception:
                    pass

                model.main(
                    margin=margin,
                    over_price_boost=over_price_boost,
                    alpha_squeeze=alpha_squeeze
                )

            # IMPORTANT: clear cached CSV loading so UI sees newest outputs
            st.cache_data.clear()
            st.success("✅ Recomputed current GW. Osvežavam rezultate…")
            st.rerun()

        df = all_gw[selected_gw]
        st.subheader(f"GW {selected_gw} – {len(df)} redova")

        # Column compatibility between v10/v11
        odds_over_col = _safe_col(df, ["bookmaker_odds_over", "odds_over"])
        odds_under_col = _safe_col(df, ["bookmaker_odds_under", "odds_under"])

        base_cols = ["home_team", "away_team", "line", "p_over", odds_over_col, odds_under_col, "mu_match"]
        extra_cols = ["is_main_line", "main_line", "p_over_absdiff_5050", "mu_home", "mu_away", "alpha_used", "mu_league"]
        cols_to_show = [c for c in base_cols + extra_cols if c and c in df.columns]

        format_dict = {
            "p_over": "{:.1%}",
            "mu_match": "{:.2f}",
            "mu_home": "{:.2f}",
            "mu_away": "{:.2f}",
            "mu_league": "{:.3f}",
            "alpha_used": "{:.4f}",
        }
        if odds_over_col:
            format_dict[odds_over_col] = "{:.2f}"
        if odds_under_col:
            format_dict[odds_under_col] = "{:.2f}"
        if "p_over_absdiff_5050" in df.columns:
            format_dict["p_over_absdiff_5050"] = "{:.3f}"

        st.dataframe(
            df[cols_to_show].style.format(format_dict),
            use_container_width=True,
            height=650
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            f"corners_gw{selected_gw}.csv",
            "text/csv"
        )


# ====================== TAB 2: CUSTOM MATCHES ======================
with tab2:
    st.subheader("🆕 Custom Matches")
    st.caption("Unesi mečeve u formatu: `Home vs Away` (po liniji).")

    custom_input = st.text_area(
        "Mečevi",
        height=180,
        placeholder="Liverpool vs Sunderland\nTottenham vs Newcastle"
    )

    # You can make lines dynamic later; for now keep a reasonable ladder
    default_lines = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]

    colA, colB = st.columns([2, 1])
    with colA:
        lines = st.multiselect("Linije", default_lines, default=default_lines)
        lines = sorted(set(lines))

    with colB:
        show_only_main = st.checkbox("Prikaži samo main line", value=False)

    if st.button("🚀 Izračunaj custom kvote", type="primary"):
        if not custom_input.strip():
            st.error("Unesi mečeve!")
        elif fitted_model is None:
            st.error("Nema fitted modela (nema model_teams_table_gw26.csv u data folderu).")
        elif not lines:
            st.error("Odaberi bar jednu liniju.")
        else:
            with st.spinner(f"Računam sa μ={current_mu:.3f} (GW{current_gw})..."):
                try:
                    model.USE_TEMPO_LAYER = use_tempo
                except Exception:
                    pass
                try:
                    model.ALPHA_SQUEEZE_DEFAULT = alpha_squeeze
                except Exception:
                    pass

                custom_rows = []
                for raw_line in custom_input.strip().split("\n"):
                    if "vs" in raw_line.lower():
                        parts = [p.strip() for p in raw_line.split("vs", 1)]
                        if len(parts) == 2:
                            home, away = [model.normalize_team_name(p) for p in parts]
                            try:
                                df_match = model.compute_fixture_odds(
                                    fitted_model,
                                    home,
                                    away,
                                    lines,
                                    margin,
                                    float(current_mu),
                                    over_price_boost,
                                    alpha_squeeze
                                )
                                # v11 returns ladder; make sure teams are present in DF
                                if "home_team" not in df_match.columns:
                                    df_match["home_team"] = home
                                if "away_team" not in df_match.columns:
                                    df_match["away_team"] = away
                                if "gameweek" not in df_match.columns:
                                    df_match["gameweek"] = 99
                                custom_rows.append(df_match)
                            except Exception as e:
                                st.error(f"Greška za {home} vs {away}: {str(e)}")

                if custom_rows:
                    custom_df = pd.concat(custom_rows, ignore_index=True)

                    # Harmonize odds column names
                    odds_over_col = _safe_col(custom_df, ["bookmaker_odds_over", "odds_over"])
                    odds_under_col = _safe_col(custom_df, ["bookmaker_odds_under", "odds_under"])

                    if show_only_main and "is_main_line" in custom_df.columns:
                        custom_df = custom_df.loc[custom_df["is_main_line"] == True].copy()

                    st.success(f"✅ Izračunato: {len(custom_df)} redova | μ={current_mu:.3f} (GW{current_gw})")

                    display_cols = ["home_team", "away_team", "line", "p_over", odds_over_col, odds_under_col, "mu_match"]
                    display_cols += [c for c in ["is_main_line", "main_line"] if c in custom_df.columns]
                    display_cols = [c for c in display_cols if c and c in custom_df.columns]

                    fmt = {
                        "p_over": "{:.1%}",
                        "mu_match": "{:.2f}",
                    }
                    if odds_over_col:
                        fmt[odds_over_col] = "{:.2f}"
                    if odds_under_col:
                        fmt[odds_under_col] = "{:.2f}"

                    st.dataframe(
                        custom_df[display_cols].style.format(fmt),
                        height=520,
                        use_container_width=True
                    )

                    csv = custom_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Custom", csv, "custom_corners.csv", "text/csv")
