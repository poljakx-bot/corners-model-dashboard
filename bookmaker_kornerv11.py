import os
import glob
import math
import logging
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# =============================================================================
# CONFIG
# =============================================================================

# --- Core features ---
USE_EMP_BIAS_CORRECTION = True
EMP_BIAS_K = 25.0
EMP_BIAS_MAX_FACTOR = 1.30

USE_TEMPO_LAYER = False
TEMPO_K = 20.0
TEMPO_MAX_FACTOR = 1.25

USE_HYBRID_INTERACTION = True
HYBRID_W = 0.62

# --- Model control ---
ALPHA_MIN, ALPHA_MAX = 0.001, 0.03
ALPHA_SQUEEZE_DEFAULT = 0.68          # možeš menjati u main()

USE_TEAM_EFFECT_CAP = True
TEAM_EFFECT_CAP = 0.55

MU_LEAGUE_ANCHOR_LOW = 0.88
MU_LEAGUE_ANCHOR_HIGH = 1.12

SOFT_ANCHOR_W_BASE = 0.08
SOFT_ANCHOR_W_LOW_SAMPLE = 0.25
SOFT_ANCHOR_W_DISTANCE = 0.12
SOFT_ANCHOR_W_UPWARD = 0.18

MISMATCH_INFLATE_ENABLED = True
MISMATCH_GAP_THRESHOLD = 0.35
MISMATCH_INFLATE_SLOPE = 0.06
MISMATCH_INFLATE_CAP = 0.10

LOW_SAMPLE_MATCHES = 15.0
UNCERTAINTY_BOOST = 0.00

OVER_PRICE_BOOST_DEFAULT = 0.00
N_SIMS = 200_000
RNG_SEED = 123



def _stable_match_seed(base_seed: int, home: str, away: str) -> int:
    """Deterministic per-fixture seed (stable across runs/OS)."""
    s = f"{int(base_seed)}|{home}|{away}".encode('utf-8', errors='ignore')
    return int(zlib.crc32(s) & 0xFFFFFFFF)
# =============================================================================
# LOGGING
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"bookmaker_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler()]
)

log = logging.getLogger("corners_model")
log.info("Logging started → %s", log_file)

# =============================================================================
# PATHS & DATA
# =============================================================================

DATA_DIR = os.path.join(BASE_DIR, "data")

ACTIVE_SEASON_HINT = "2025-to-2026"

SEASON_WEIGHTS = {
    "2023-to-2024": 0.0,
    "2024-to-2025": 1.0,
    "2025-to-2026": 1.0,
}

# Regularizacija
L2_LAMBDA_TEAM_BASE = 0.06
L2_LAMBDA_HOMEADV   = 0.01
EXPOSURE_EPS        = 5.0

HALF_LIFE_GW = 5.0
SEASON_GW = 38

# FootyStats columns
COL_HOME_TEAM = "home_team_name"
COL_AWAY_TEAM = "away_team_name"
COL_HOME_CORNERS = "home_team_corner_count"
COL_AWAY_CORNERS = "away_team_corner_count"
COL_STATUS = "status"
COL_GAMEWEEK = "Game Week"

COL_HOME_GOALS = "home_team_goal_count"
COL_AWAY_GOALS = "away_team_goal_count"

COMPLETED_STATUSES = {"complete", "finished", "ft", "ended"}
DEFAULT_LINES = [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
MATCH_PATTERN_PL = "*premier-league-matches*-stats*.csv"

# =============================================================================
# BOOKMAKER ODDS
# =============================================================================

def bookmaker_odds_two_way(p_over: float, margin: float, over_price_boost: float = 0.0):
    p_over = float(np.clip(p_over, 1e-9, 1.0 - 1e-9))
    p_under = 1.0 - p_over
    target = 1.0 + float(margin)

    imp_over = p_over * target
    imp_under = p_under * target

    if over_price_boost != 0.0:
        shift = float(over_price_boost) * imp_over
        imp_over = max(1e-9, imp_over - shift)
        imp_under = max(1e-9, imp_under + shift)

    s = imp_over + imp_under
    if s > 0:
        scale = target / s
        imp_over *= scale
        imp_under *= scale

    return 1.0 / imp_over, 1.0 / imp_under, imp_over, imp_under


# =============================================================================
# HELPERS
# =============================================================================

def normalize_team_name(x: str) -> str:
    s = str(x).strip()
    s = " ".join(s.split()).lower()
    mapping = {
        "leeds": "Leeds United", "leeds united": "Leeds United",
        "afc bournemouth": "Bournemouth", "bournemouth": "Bournemouth",
        "tottenham hotspur": "Tottenham", "tottenham": "Tottenham",
        "newcastle united": "Newcastle", "newcastle": "Newcastle",
        "leicester city": "Leicester", "leicester": "Leicester",
        "nottingham forest": "Nottingham Forest",
        "southampton": "Southampton", "everton": "Everton",
        "burnley": "Burnley", "sunderland": "Sunderland",
    }
    return mapping.get(s, s.capitalize())


def _find_pl_season_csv(season_hint: str) -> str | None:
    patterns = [
        os.path.join(DATA_DIR, f"*premier-league-matches-{season_hint}-stats*.csv"),
        os.path.join(BASE_DIR, f"*premier-league-matches-{season_hint}-stats*.csv"),
        os.path.join(os.getcwd(), f"*premier-league-matches-{season_hint}-stats*.csv"),
    ]
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return sorted(hits)[0]
    return None


# =============================================================================
# NEGATIVE BINOMIAL
# =============================================================================

def nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    r = 1.0 / alpha
    p = r / (r + mu)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1) +
            r * np.log(p) + y * np.log(1.0 - p))


def nb_rvs_vectorized(mu: float, alpha: float, rng: np.random.Generator, size: int) -> np.ndarray:
    """Safe NB simulacija – clip za ekstreme."""
    r = 1.0 / alpha
    scale = mu / r
    lam = rng.gamma(shape=r, scale=scale, size=size)
    
    # CRUCIAL: Spreči too large / NaN / inf
    lam = np.clip(lam, 1e-8, 5000.0)  # 5000 je više nego dovoljno za corners (max realno ~40)
    
    return rng.poisson(lam)


# =============================================================================
# DATA LOADING & WEIGHTING
# =============================================================================

def find_match_files(data_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"Nema fajlova sa pattern-om: {pattern}")
    return files


def load_matches(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in [COL_HOME_TEAM, COL_AWAY_TEAM, COL_HOME_CORNERS, COL_AWAY_CORNERS]:
        if col not in df.columns:
            raise ValueError(f"Fali kolona {col} u {csv_path}")

    df["_status_l"] = df.get(COL_STATUS, "complete").astype(str).str.lower().str.strip()
    df["_gameweek"] = pd.to_numeric(df.get(COL_GAMEWEEK), errors="coerce")

    for col in [COL_HOME_CORNERS, COL_AWAY_CORNERS, COL_HOME_GOALS, COL_AWAY_GOALS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[COL_HOME_TEAM] = df[COL_HOME_TEAM].map(normalize_team_name)
    df[COL_AWAY_TEAM] = df[COL_AWAY_TEAM].map(normalize_team_name)

    return df


def filter_completed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["_status_l"].isin(COMPLETED_STATUSES)].dropna(
        subset=[COL_HOME_CORNERS, COL_AWAY_CORNERS]
    )


def season_weight_from_filename(path: str) -> float:
    b = os.path.basename(path)
    return next((w for k, w in SEASON_WEIGHTS.items() if k in b), 1.0)


def _season_order_index(path: str) -> int:
    b = os.path.basename(path)
    keys = list(SEASON_WEIGHTS.keys())
    return next((i for i, k in enumerate(keys) if k in b), len(keys) - 1)


def _compute_recency_decay_weight(age_gw: float, half_life: float) -> float:
    return 0.5 ** (max(float(age_gw), 0.0) / float(half_life)) if half_life > 0 else 1.0


def build_weighted_training_df(files: List[str], current_t_gw: float) -> pd.DataFrame:
    frames = []
    for f in files:
        df = filter_completed(load_matches(f))
        if df.empty:
            continue

        df["_w"] = season_weight_from_filename(f)
        season_idx = _season_order_index(f)
        t_match = season_idx * SEASON_GW + df["_gameweek"].fillna(0)
        age = float(current_t_gw) - t_match.astype(float)
        df["_w"] *= age.apply(lambda x: _compute_recency_decay_weight(x, HALF_LIFE_GW))
        df["_source_file"] = os.path.basename(f)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# MODEL
# =============================================================================

@dataclass
class FittedModel:
    teams: List[str]
    beta0: float
    home_adv: float
    alpha: float
    attack: Dict[str, float]
    defense: Dict[str, float]
    matches_w: Dict[str, float]
    team_tempo_factor: Dict[str, float] = field(default_factory=dict)
    emp_against_factor: Dict[str, float] = field(default_factory=dict)


def _cap_and_recenter(vec: np.ndarray, cap: float) -> np.ndarray:
    """Clip vector to [-cap, cap] then re-center to sum to ~0."""
    v = np.clip(vec, -float(cap), float(cap)).astype(float)
    v = v - float(np.mean(v))
    return v


def build_team_index(df: pd.DataFrame) -> List[str]:
    return sorted(set(df[COL_HOME_TEAM]) | set(df[COL_AWAY_TEAM]))


def initial_params(df: pd.DataFrame, teams: List[str]):
    mean_total = float(df[COL_HOME_CORNERS].mean() + df[COL_AWAY_CORNERS].mean())
    league_per_team = mean_total / 2

    beta0 = math.log(league_per_team)
    home_adv = math.log(max(df[COL_HOME_CORNERS].mean(), 1e-6)) - math.log(max(df[COL_AWAY_CORNERS].mean(), 1e-6))

    alpha = 0.012

    home_for = df.groupby(COL_HOME_TEAM)[COL_HOME_CORNERS].mean()
    away_for = df.groupby(COL_AWAY_TEAM)[COL_AWAY_CORNERS].mean()
    home_against = df.groupby(COL_HOME_TEAM)[COL_AWAY_CORNERS].mean()
    away_against = df.groupby(COL_AWAY_TEAM)[COL_HOME_CORNERS].mean()

    a0 = [math.log(max(home_for.get(t, league_per_team), 1e-6) / league_per_team) for t in teams]
    d0 = [math.log(max(home_against.get(t, league_per_team), 1e-6) / league_per_team) for t in teams]

    a0 = np.array(a0) - np.mean(a0)
    d0 = np.array(d0) - np.mean(d0)

    return beta0, home_adv, alpha, a0, d0


def pack_params(beta0, home_adv, alpha, attack, defense, n):
    return np.concatenate(([beta0, home_adv, math.log(alpha)], attack[:-1], defense[:-1]))


def unpack_params(theta: np.ndarray, n: int):
    beta0 = float(theta[0])
    home_adv = float(theta[1])
    alpha = float(np.exp(theta[2]))

    a = np.zeros(n)
    d = np.zeros(n)
    a[:-1] = theta[3:3 + n - 1]
    d[:-1] = theta[3 + n - 1:]
    a[-1] = -a[:-1].sum()
    d[-1] = -d[:-1].sum()

    if USE_TEAM_EFFECT_CAP:
        a = _cap_and_recenter(a, TEAM_EFFECT_CAP)
        d = _cap_and_recenter(d, TEAM_EFFECT_CAP)

    return beta0, home_adv, alpha, a, d


def compute_team_exposure_weighted(df: pd.DataFrame, teams: List[str]) -> Dict[str, float]:
    w = df.get("_w", 1.0).astype(float)
    home_exp = df.groupby(COL_HOME_TEAM).apply(lambda g: w.loc[g.index].sum())
    away_exp = df.groupby(COL_AWAY_TEAM).apply(lambda g: w.loc[g.index].sum())
    return {t: float(home_exp.get(t, 0) + away_exp.get(t, 0)) for t in teams}


def fit_negbin_team_model_weighted(df: pd.DataFrame) -> FittedModel:
    teams = build_team_index(df)
    n = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    matches_w = compute_team_exposure_weighted(df, teams)
    exp_vec = np.array([matches_w[t] for t in teams])
    lam_vec = L2_LAMBDA_TEAM_BASE * (exp_vec.mean() / (exp_vec + EXPOSURE_EPS))
    lam_vec = np.clip(lam_vec, 0.05 * L2_LAMBDA_TEAM_BASE, 5.0 * L2_LAMBDA_TEAM_BASE)

    beta0, home_adv, alpha, a0, d0 = initial_params(df, teams)
    theta0 = pack_params(beta0, home_adv, alpha, a0, d0, n)

    h_idx = df[COL_HOME_TEAM].map(team_to_idx).to_numpy()
    a_idx = df[COL_AWAY_TEAM].map(team_to_idx).to_numpy()
    y_h = df[COL_HOME_CORNERS].round().astype(int).clip(lower=0).to_numpy()
    y_a = df[COL_AWAY_CORNERS].round().astype(int).clip(lower=0).to_numpy()
    w = df.get("_w", 1.0).astype(float).to_numpy()

    def nll(theta):
        b0, hadv, alp, A, D = unpack_params(theta, n)
        if not (ALPHA_MIN <= alp <= ALPHA_MAX):
            return 1e18

        mu_h = np.exp(b0 + hadv + A[h_idx] + D[a_idx])
        mu_a = np.exp(b0 + A[a_idx] + D[h_idx])

        ll = nb_logpmf(y_h, mu_h, alp) + nb_logpmf(y_a, mu_a, alp)
        return -float(np.sum(w * ll)) + float(np.sum(lam_vec * (A**2 + D**2))) + L2_LAMBDA_HOMEADV * hadv**2

    res = minimize(nll, theta0, method="L-BFGS-B",
                   bounds=[(None, None)] * 2 + [(math.log(ALPHA_MIN), math.log(ALPHA_MAX))] + [(None, None)] * (2 * (n - 1)),
                   options={"maxiter": 2500})

    beta0_f, home_adv_f, alpha_f, A_f, D_f = unpack_params(res.x, n)

    # Tempo layer
    team_tempo_factor = {}
    if USE_TEMPO_LAYER:
        tot = (df[COL_HOME_CORNERS] + df[COL_AWAY_CORNERS]).fillna(0)
        w_s = df.get("_w", 1.0)
        league_mean = float((tot * w_s).sum() / w_s.sum())

        for t in teams:
            mask = (df[COL_HOME_TEAM] == t) | (df[COL_AWAY_TEAM] == t)
            tot_t = tot[mask]
            w_t = w_s[mask]
            n_eff = float(w_t.sum())
            mean_t = float((tot_t * w_t).sum() / w_t.sum()) if n_eff > 0 else league_mean
            raw = mean_t / league_mean
            shrink = n_eff / (n_eff + TEMPO_K)
            fac = 1.0 + shrink * (raw - 1.0)
            fac = np.clip(fac, 1.0 / TEMPO_MAX_FACTOR, TEMPO_MAX_FACTOR)
            team_tempo_factor[t] = float(fac)


    # -------------------------------------------------------------------------
    # Empirical 'against' bias correction
    # For each opponent team T, compute factor = empirical_conceded(T) / model_pred_conceded(T)
    # Then, when predicting corners FOR a team vs T, multiply mu by this factor.
    # Uses shrinkage (EMP_BIAS_K) and caps (EMP_BIAS_MAX_FACTOR).
    # -------------------------------------------------------------------------
    emp_against_factor = {}
    if USE_EMP_BIAS_CORRECTION:
        w = df.get('_w', 1.0).astype(float)
        emp_conc_sum = {t: 0.0 for t in teams}
        emp_conc_w = {t: 0.0 for t in teams}
        pred_conc_sum = {t: 0.0 for t in teams}
        pred_conc_w = {t: 0.0 for t in teams}

        for i, r in df.iterrows():
            h = r[COL_HOME_TEAM]
            a = r[COL_AWAY_TEAM]
            wi = float(w.loc[i])

            # empirical conceded
            emp_conc_sum[h] += wi * float(r[COL_AWAY_CORNERS])
            emp_conc_w[h] += wi
            emp_conc_sum[a] += wi * float(r[COL_HOME_CORNERS])
            emp_conc_w[a] += wi

            # predicted conceded (base model)
            ih = team_to_idx[h]
            ia = team_to_idx[a]
            mu_h_pred = math.exp(beta0_f + home_adv_f + A_f[ih] + D_f[ia])
            mu_a_pred = math.exp(beta0_f + A_f[ia] + D_f[ih])
            pred_conc_sum[h] += wi * mu_a_pred
            pred_conc_w[h] += wi
            pred_conc_sum[a] += wi * mu_h_pred
            pred_conc_w[a] += wi

        for t in teams:
            if emp_conc_w[t] <= 0 or pred_conc_w[t] <= 0:
                emp_against_factor[t] = 1.0
                continue
            emp_mean = emp_conc_sum[t] / emp_conc_w[t]
            pred_mean = max(pred_conc_sum[t] / pred_conc_w[t], 1e-9)
            raw = emp_mean / pred_mean
            n_eff = float(matches_w.get(t, 0.0))
            shrink = n_eff / (n_eff + EMP_BIAS_K)
            fac = 1.0 + shrink * (raw - 1.0)
            fac = float(np.clip(fac, 1.0 / EMP_BIAS_MAX_FACTOR, EMP_BIAS_MAX_FACTOR))
            emp_against_factor[t] = fac
    return FittedModel(
        teams=teams,
        beta0=beta0_f,
        home_adv=home_adv_f,
        alpha=alpha_f,
        attack={t: float(A_f[i]) for i, t in enumerate(teams)},
        defense={t: float(D_f[i]) for i, t in enumerate(teams)},
        matches_w=matches_w,
        team_tempo_factor=team_tempo_factor,
        emp_against_factor=emp_against_factor
    )


def export_model_tables(model: FittedModel, df_train: pd.DataFrame, out_dir: str, tag: str = "") -> tuple[str, str]:
    """Export Teams and Sanity tables for quick model/data verification."""
    os.makedirs(out_dir, exist_ok=True)

    teams = sorted(set(model.attack.keys()) | set(model.defense.keys()))

    rows = []
    for t in teams:
        att = float(model.attack.get(t, 0.0))
        deff = float(model.defense.get(t, 0.0))
        tempo = float(model.team_tempo_factor.get(t, 1.0))
        mw = float(model.matches_w.get(t, np.nan))

        mu_home_vs_avg = float(np.exp(model.beta0 + model.home_adv + att + np.mean(list(model.defense.values())))) * tempo
        mu_away_vs_avg = float(np.exp(model.beta0 + att + np.mean(list(model.defense.values())))) * tempo

        rows.append({
            "team": t,
            "matches_w": mw,
            "tempo": tempo,
            "attack": att,
            "defense": deff,
            "mu_home_vs_avg": mu_home_vs_avg,
            "mu_away_vs_avg": mu_away_vs_avg,
            "mu_total_vs_avg": mu_home_vs_avg + mu_away_vs_avg,
        })

    df_teams = pd.DataFrame(rows).sort_values(["matches_w", "team"])

    # Sanity table
    total_c = (pd.to_numeric(df_train[COL_HOME_CORNERS], errors="coerce").fillna(0) +
               pd.to_numeric(df_train[COL_AWAY_CORNERS], errors="coerce").fillna(0))
    emp_mean = float(total_c.mean())
    emp_std = float(total_c.std(ddof=0))

    sanity_rows = [
        {"metric": "train_rows", "value": float(len(df_train))},
        {"metric": "emp_total_corners_mean", "value": emp_mean},
        {"metric": "emp_total_corners_std", "value": emp_std},
        {"metric": "beta0", "value": float(model.beta0)},
        {"metric": "home_adv", "value": float(model.home_adv)},
        {"metric": "alpha", "value": float(model.alpha)},
        {"metric": "tempo_min", "value": float(min(model.team_tempo_factor.values())) if model.team_tempo_factor else np.nan},
        {"metric": "tempo_max", "value": float(max(model.team_tempo_factor.values())) if model.team_tempo_factor else np.nan},
    ]
    df_sanity = pd.DataFrame(sanity_rows)

    suf = f"_{tag}" if tag else ""
    teams_path = os.path.join(out_dir, f"model_teams_table{suf}.csv")
    sanity_path = os.path.join(out_dir, f"model_sanity_table{suf}.csv")
    df_teams.to_csv(teams_path, index=False, encoding="utf-8")
    df_sanity.to_csv(sanity_path, index=False, encoding="utf-8")
    return teams_path, sanity_path


# =============================================================================
# SIMULATION
# =============================================================================

def is_low_sample_team(model: FittedModel, team: str) -> bool:
    return model.matches_w.get(team, 0.0) < LOW_SAMPLE_MATCHES


def alpha_for_match(model: FittedModel, home: str, away: str, alpha_squeeze: float) -> float:
    a = float(model.alpha) * float(alpha_squeeze)
    a = np.clip(a, ALPHA_MIN, ALPHA_MAX)
    if is_low_sample_team(model, home) or is_low_sample_team(model, away):
        a *= (1.0 + UNCERTAINTY_BOOST)
    return float(np.clip(a, ALPHA_MIN, ALPHA_MAX))


def simulate_total_corners(
    model: FittedModel,
    home: str,
    away: str,
    n_sims: int,
    mu_pl_league_match: float,
    alpha_squeeze: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:


    # Base model
    mu_h = math.exp(model.beta0 + model.home_adv + model.attack[home] + model.defense[away])
    mu_a = math.exp(model.beta0 + model.attack[away] + model.defense[home])

    # Hybrid interaction
    if USE_HYBRID_INTERACTION:
        att_h = math.exp(model.beta0 + model.home_adv + model.attack[home])
        vul_a = math.exp(model.beta0 + model.defense[away])
        mu_h_add = 0.5 * (att_h + vul_a)

        att_a = math.exp(model.beta0 + model.attack[away])
        vul_h = math.exp(model.beta0 + model.defense[home])
        mu_a_add = 0.5 * (att_a + vul_h)

        mu_h = HYBRID_W * mu_h + (1 - HYBRID_W) * mu_h_add
        mu_a = HYBRID_W * mu_a + (1 - HYBRID_W) * mu_a_add

    # Empirical bias correction
    if USE_EMP_BIAS_CORRECTION and hasattr(model, "emp_against_factor"):
        mu_h *= model.emp_against_factor.get(away, 1.0)
        mu_a *= model.emp_against_factor.get(home, 1.0)

    mu_h_adj = float(mu_h)
    mu_a_adj = float(mu_a)

    # Mismatch inflation
    if MISMATCH_INFLATE_ENABLED:
        gap = abs(model.attack.get(home, 0) - model.attack.get(away, 0))
        if gap > MISMATCH_GAP_THRESHOLD:
            infl = 1.0 + min(MISMATCH_INFLATE_CAP, MISMATCH_INFLATE_SLOPE * (gap - MISMATCH_GAP_THRESHOLD))
            mu_h_adj *= infl
            mu_a_adj *= infl
    # ------------------------------------------------------------------
    # CHAOS RELEASE (EPL survival-style teams don't suppress corners)
    # This corrects NB bias where low-possession teams look like low-event.
    # ------------------------------------------------------------------
    away_att = model.attack.get(away, 0.0)
    away_def = model.defense.get(away, 0.0)

    home_att = model.attack.get(home, 0.0)
    home_def = model.defense.get(home, 0.0)

    # detect low-possession away side (typical relegation profile)
    if away_att < -0.18 and away_def > 0:
        chaos_factor = 1.05
    # mirror case (rare but symmetric)
    elif home_att < -0.18 and home_def > 0:
        chaos_factor = 1.04
    else:
        chaos_factor = 1.00

    mu_h_adj *= chaos_factor
    mu_a_adj *= chaos_factor

    # Tempo + soft anchor
    mu_total = mu_h_adj + mu_a_adj
    mu_pl_anchor = float(mu_pl_league_match)

    if USE_TEMPO_LAYER and model.team_tempo_factor:
        th = model.team_tempo_factor.get(home, 1.0)
        ta = model.team_tempo_factor.get(away, 1.0)
        tempo = math.sqrt(th * ta)
        mu_total *= tempo
        mu_pl_anchor *= tempo

    # Soft anchor
    low_sample = is_low_sample_team(model, home) or is_low_sample_team(model, away)
    if mu_pl_anchor > 0:
        dist = abs(mu_total - mu_pl_anchor) / mu_pl_anchor
        w = SOFT_ANCHOR_W_LOW_SAMPLE if low_sample else SOFT_ANCHOR_W_BASE
        w += SOFT_ANCHOR_W_DISTANCE * min(dist, 1.0)
        if mu_total < mu_pl_anchor:
            w += SOFT_ANCHOR_W_UPWARD * min((mu_pl_anchor - mu_total) / mu_pl_anchor, 1.0)
        w = min(w, 0.85)
        mu_total = (1 - w) * mu_total + w * mu_pl_anchor

    mu_total = np.clip(mu_total,
                       MU_LEAGUE_ANCHOR_LOW * mu_pl_anchor,
                       MU_LEAGUE_ANCHOR_HIGH * mu_pl_anchor)

    # Split back
    ratio_h = np.clip(mu_h_adj / (mu_h_adj + mu_a_adj), 0.30, 0.70)
    mu_h_final = mu_total * ratio_h
    mu_a_final = mu_total * (1 - ratio_h)

    # ROBUSTNOST: Nikad ne dozvoli ekstremne mu (corners ne idu iznad 40 po timu)
    mu_h_final = float(np.clip(mu_h_final, 0.01, 40.0))
    mu_a_final = float(np.clip(mu_a_final, 0.01, 40.0))

    # Simulation
    a_match = alpha_for_match(model, home, away, alpha_squeeze)
    sim_h = nb_rvs_vectorized(mu_h_final, a_match, rng, n_sims)
    sim_a = nb_rvs_vectorized(mu_a_final, a_match, rng, n_sims)
    sim_t = sim_h + sim_a

    return sim_h, sim_a, sim_t, mu_h_final, mu_a_final, a_match


# =============================================================================
# ODDS COMPUTATION
# =============================================================================

def compute_fixture_odds(
    model: FittedModel,
    home: str,
    away: str,
    lines: List[float],
    margin: float,
    mu_pl_league_match: float,
    over_price_boost: float = OVER_PRICE_BOOST_DEFAULT,
    alpha_squeeze: float = ALPHA_SQUEEZE_DEFAULT
) -> pd.DataFrame:

    # deterministic per-fixture RNG (avoids same stream for every match)
    seed = _stable_match_seed(RNG_SEED, home, away)
    rng = np.random.default_rng(seed)

    sim_h, sim_a, sim_t, mu_h, mu_a, a_match = simulate_total_corners(
        model, home, away, N_SIMS, mu_pl_league_match, alpha_squeeze, rng
    )

    rows = []
    best = None
    best_diff = float('inf')

    for line in lines:
        p_over = float(np.mean(sim_t > line))
        diff_5050 = abs(p_over - 0.5)

        book_over, book_under, _, _ = bookmaker_odds_two_way(p_over, margin, over_price_boost)

        row = {
            "home": home,
            "away": away,
            "market": "corners_total",
            "line": line,
            "p_over": p_over,
            "p_under": 1 - p_over,
            "bookmaker_odds_over": book_over,
            "bookmaker_odds_under": book_under,
            "mu_match": mu_h + mu_a,
            "mu_home": mu_h,
            "mu_away": mu_a,
            "mu_league": mu_pl_league_match,
            "alpha_base": float(model.alpha),
            "alpha_squeeze": float(alpha_squeeze),
            "alpha_used": float(a_match),
            "p_over_absdiff_5050": diff_5050,
        }

        rows.append(row)

        if diff_5050 < best_diff:
            best = row
            best_diff = diff_5050

    df_out = pd.DataFrame(rows)
    if best is not None and not df_out.empty:
        df_out["is_main_line"] = df_out["line"].astype(float).eq(float(best["line"]))
        df_out["main_line"] = float(best["line"])
    else:
        df_out["is_main_line"] = False
        df_out["main_line"] = np.nan
    return df_out


# =============================================================================
# MAIN
# =============================================================================

def main(
    margin: float = 0.08,           # SHARP_MARGIN
    over_price_boost: float = 0.018, # OVER_PRICE_BOOST_SHARP
    alpha_squeeze: float = ALPHA_SQUEEZE_DEFAULT
):
    pl_files = find_match_files(DATA_DIR, MATCH_PATTERN_PL)
    active_path = next(f for f in pl_files if ACTIVE_SEASON_HINT in f)

    df_active = load_matches(active_path)

    # Robusnije računanje current gameweek
    played = df_active[df_active["_status_l"].isin(COMPLETED_STATUSES)]
    current_gw = int(played[COL_GAMEWEEK].max()) + 1 if not played.empty else 1

    train_files = [f for f in pl_files if any(k in f for k in SEASON_WEIGHTS)]
    current_t_gw = _season_order_index(active_path) * SEASON_GW + current_gw
    df_train = build_weighted_training_df(train_files, current_t_gw)  # ← OVO PRVO!

    log.info("Trening: %d utakmica (do GW %d)", len(df_train), current_gw - 1)

    model = fit_negbin_team_model_weighted(df_train)

    # FIX ZA EARLY GW (GW1): Ako nema played mečeva, koristi train mean
    if len(played) == 0 or pd.isna((played[COL_HOME_CORNERS] + played[COL_AWAY_CORNERS]).mean()):
        mu_league = float((df_train[COL_HOME_CORNERS] + df_train[COL_AWAY_CORNERS]).mean())
        log.warning("No played matches in active season → using train mean μ=%.3f", mu_league)
    else:
        mu_league = float((played[COL_HOME_CORNERS] + played[COL_AWAY_CORNERS]).mean())
        log.info("μ league (trenutni prosek) = %.3f", mu_league)

    # Export tabela za proveru
    teams_path, sanity_path = export_model_tables(model, df_train, DATA_DIR, tag=f"gw{current_gw}")
    log.info("[OK] teams table: %s", teams_path)
    log.info("[OK] sanity table: %s", sanity_path)

    # Sledeći GW
    df_next = df_active[pd.to_numeric(df_active[COL_GAMEWEEK], errors="coerce") == current_gw]

    all_rows = []
    for _, r in df_next.iterrows():
        h = normalize_team_name(r[COL_HOME_TEAM])
        a = normalize_team_name(r[COL_AWAY_TEAM])

        if h not in model.attack or a not in model.attack:
            log.warning("Nedostaje tim: %s ili %s", h, a)
            continue

        # Dinamičke linije oko očekivanog proseka
        mu_est = 9.8
        base_line = round(mu_est * 2) / 2 - 0.5
        lines = [base_line - 2, base_line - 1, base_line, base_line + 1, base_line + 2, base_line + 3]

        df_match = compute_fixture_odds(
            model, h, a, lines, margin, mu_league, over_price_boost, alpha_squeeze
        )
        df_match = df_match.assign(gameweek=current_gw, home_team=h, away_team=a)
        all_rows.append(df_match)

    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(DATA_DIR, f"bookmaker_odds_corners_gw{current_gw}_margin{margin:.3f}.csv")
    out.to_csv(out_path, index=False)
    log.info("Spremljeno → %s (%d redova)", out_path, len(out))

if __name__ == "__main__":
    main()