import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
import json
np.set_printoptions(legacy='1.25')

# --- Utility ---
spot_cases=[-3,-2,-1,0,1,2,3]
def parse_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

# def quadratic(x, a, b, c):
#     return a * x**2 + b * x + c



def black_scholes_with_greeks(S, K, sigma, t, option_type, iv_cases, spot_cases=[-3,-2,-1,0,1,2,3], dte_cases=[0, -1, -3, -7, -10, -14],r=.0433):
    if t <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    # print(iv_cases)
    d1 = (math.log(S / K) + (0.5 * sigma ** 2 +r)* t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    multiplier = 1000 / 15.625
    
    S_cases=[S+spot_case for spot_case in spot_cases]
    T_cases=[max(t+dte_case/365, 0.0001) for dte_case in dte_cases]
    print(t*365,365*np.array(T_cases))
    d1_cases = [(math.log(Sc / K) + (0.5 * iv_cases[i] ** 2+ r) * t) / (iv_cases[i] * math.sqrt(t)) for i, Sc in enumerate(S_cases)] #uses vol cases also for scenario analysis
    #d1_cases = [(math.log(Sc / K) + 0.5 * sigma ** 2 * t) / (sigma * math.sqrt(t)) for Sc in S_cases]
    #d2_cases = [d1c - sigma * math.sqrt(t) for d1c in d1_cases]
    d2_cases = [d1c - iv_cases[i] * math.sqrt(t) for i, d1c in enumerate(d1_cases)] #uses vol cases also for scenario analysis
    # d1_t_cases=[(math.log(S / K) + (0.5 * sigma ** 2 +r)* t_case) / (sigma * math.sqrt(t_case)) for t_case in T_cases]
    # d2_t_cases=[d1_t_case- sigma * math.sqrt(t_case) for i,(t_case,d1_t_case) in enumerate(zip(T_cases), d1_t_cases)]

    if option_type == "Call":
        price = multiplier * (S * norm.cdf(d1) - math.exp(-r*t)*K * norm.cdf(d2))
        # price_t_cases = [multiplier * (S * norm.cdf(d1_t_case) - math.exp(-r*t_case)*K * norm.cdf(d2_t_case)) for d1_t_case, d2_t_case, t_case in zip(d1_t_cases, d2_t_cases, T_cases)]
        spot_cases = [multiplier * (Sc * norm.cdf(d1c) - math.exp(-r*t)*K * norm.cdf(d2c)) for Sc, d1c, d2c in zip(S_cases, d1_cases, d2_cases)]
        delta = norm.cdf(d1)
        theta = (-S * sigma * norm.pdf(d1)) / (2 * math.sqrt(t))-r*K*math.exp(-r*t)*norm.cdf(d2)
        d_cases = [norm.cdf(d1c) for d1c in d1_cases]
        t_cases = [(-Sc * iv_cases[i] * norm.pdf(d1c)) / (2 * math.sqrt(t))-r*K*math.exp(-r*t)*norm.cdf(d2c) for i, (Sc, d1c,d2c) in enumerate(zip(S_cases, d1_cases, d2_cases))]
    elif option_type == "Put":
        price = multiplier * (math.exp(-r*t)*K * norm.cdf(-d2) - S * norm.cdf(-d1))
        #price_t_cases = [multiplier * (math.exp(-r*t_case)*K * norm.cdf(-d2_t_case) - S * norm.cdf(-d1_t_case)) for d1_t_case, d2_t_case, t_case in zip(d1_t_cases, d2_t_cases, T_cases)]
        spot_cases = [multiplier * (math.exp(-r*t)*K * norm.cdf(-d2c) - Sc * norm.cdf(-d1c)) for Sc, d1c, d2c in zip(S_cases, d1_cases, d2_cases)]
        delta = norm.cdf(d1) - 1
        theta = (-S * sigma * norm.pdf(d1)) / (2 * math.sqrt(t))-r*K*math.exp(-r*t)*norm.cdf(-d2)
        d_cases= [(norm.cdf(d1c)) for d1c in d1_cases]
        t_cases = [(-Sc * iv_cases[i] * norm.pdf(d1c)) / (2 * math.sqrt(t))-r*K*math.exp(-r*t)*norm.cdf(-d2c) for i, (Sc, d1c,d2c) in enumerate(zip(S_cases, d1_cases, d2_cases))]
    else:
        return None

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(t))
    vega = (S * norm.pdf(d1) * math.sqrt(t)) / 100
    g_cases = [norm.pdf(d1c) / (Sc * iv_cases[i] * math.sqrt(t)) if Sc > 0 and iv_cases[i] > 0 else None for i, (Sc, d1c) in enumerate(zip(S_cases, d1_cases))]
    v_cases = [(Sc * norm.pdf(d1c) * math.sqrt(t)) / 100 if Sc > 0 else None for Sc, d1c in zip(S_cases, d1_cases)]
    #spot scenarios for different dte scenarios 
    spot_t_cases = []
    for t_c in T_cases:
        d1_cases = [(math.log(Sc / K) + (0.5 * iv_cases[i] ** 2+ r) * t_c) / (iv_cases[i] * math.sqrt(t_c)) for i, Sc in enumerate(S_cases)]
        d2_cases = [d1c - iv_cases[i] * math.sqrt(t_c) for i, d1c in enumerate(d1_cases)]
        if option_type == "Call":
            spot_t_cases += [multiplier * (Sc * norm.cdf(d1c) - math.exp(-r*t_c)*K * norm.cdf(d2c)) for Sc, d1c, d2c in zip(S_cases, d1_cases, d2_cases)]
        elif option_type == "Put":
            spot_t_cases += [multiplier * (math.exp(-r*t_c)*K * norm.cdf(-d2c) - Sc * norm.cdf(-d1c)) for Sc, d1c, d2c in zip(S_cases, d1_cases, d2_cases)]
    spot_t_cases = np.array(spot_t_cases).reshape(len(T_cases), len(spot_cases)).tolist()  # Reshape to 2D list
    return {
        "Premium": price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "s_cases": spot_cases,
        "t_cases_curr": spot_t_cases[0],
        "t_cases_1day": spot_t_cases[1],
        "t_cases_3day": spot_t_cases[2],
        "t_cases_7day": spot_t_cases[3],
        "t_cases_10day": spot_t_cases[4],
        "t_cases_14day": spot_t_cases[5],
        'T_cases': T_cases,
        # "price_t_cases": price_t_cases,
        "t_cases": t_cases,
        "d_cases": d_cases,
        "g_cases": g_cases,
        "v_cases": v_cases
    }

def iv_from_premium(S, K, premium, orig_iv, t, option_type, iv_cases, spot_cases=[-3,-2,-1,0,1,2,3]): #using binary search method
    low = 0.0
    high = 60.0
    accuracy = 0.01
    iv = None

    while high - low > accuracy:
        mid = (low + high) / 2
        result = black_scholes_with_greeks(S, K, mid, t, option_type, iv_cases, spot_cases)
        if result is None:
            return None
        calc_premium = result["Premium"]
        if calc_premium > premium:
            high = mid
        else:
            low = mid

    iv = (low + high) / 2
    # Calculate final premium difference
    final_result = black_scholes_with_greeks(S, K, iv, t, option_type, iv_cases, spot_cases)
    if final_result is not None:
        premium_diff = final_result["Premium"] - premium
        premium_diff_model=black_scholes_with_greeks(S, K, iv_cases[3], t, option_type, iv_cases, spot_cases)['Premium'] -premium
        print(f"Strike: {K:.4f}, DTE: {t*365},IV found: {iv:.4f}, IV orig: {orig_iv:.4f}, synth_iv: {iv_cases[3]:.4f}, Prem_diff: {premium_diff:.4f}, Prem_diff_model: {premium_diff_model:.4f}")
    return iv

def newton_iv_from_premium(S, K, premium, orig_iv, t, option_type, iv_cases, spot_cases=[-3,-2,-1,0,1,2,3]): #using binary search method
    # Use Newton-Ralphson method for faster convergence
    iv = orig_iv if orig_iv is not None and orig_iv > 0 else 1.0
    max_iter = 100
    tol = 1e-4

    for _ in range(max_iter):
        result = black_scholes_with_greeks(S, K, iv, t, option_type, iv_cases, spot_cases)
        if result is None:
            return None
        price = result["Premium"]
        vega = result["Vega"] * 100  # Undo /100 in black_scholes_with_greeks
        diff = price - premium
        if abs(diff) < tol:
            break
        if vega == 0 or np.isnan(vega):
            break
        iv -= diff / vega
        if iv <= 0:
            iv = tol  # Prevent negative or zero IV

    # Calculate final premium difference
    final_result = black_scholes_with_greeks(S, K, iv, t, option_type, iv_cases, spot_cases)
    if final_result is not None:
        premium_diff = final_result["Premium"] - premium
        premium_diff_model = black_scholes_with_greeks(S, K, iv_cases[3], t, option_type, iv_cases, spot_cases)['Premium'] - premium
        print(f"Strike: {K:.4f}, DTE: {t*365},IV found: {iv:.4f}, IV orig: {orig_iv:.4f}, synth_iv: {iv_cases[3]:.4f}, Prem_diff: {premium_diff:.4f}, Prem_diff_model: {premium_diff_model:.4f}")
    return iv

month_map = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
}

def split_mixed_add(num):
    integer_part = int(num)
    decimal_part = num - integer_part
    result = (integer_part * 64) + (decimal_part)*100
    return result

def convert_hyphen_code(code):
    # Split on hyphen
    try:
        month_str, year_suffix = code.split('-')
        month_str = month_str[:3].lower()  # normalize to 3-letter capitalized
        year = 2000 + int(year_suffix)
        if month_str not in month_map:
            raise ValueError(f"Unknown month prefix: {month_str}")
        return f"{year}{month_map[month_str]}"
    except Exception as e:
        return f"Error: {e}"

def convert_code_to_yyyymm(code):
    """Convert codes like 'Jun25' to '202506'"""
    month_str = code[:3].lower()
    year_suffix = int(code[3:])
    year = 2000 + year_suffix  # Assuming all years are 2000+
    month = month_map.get(month_str, '00')
    return f"{year}{month}"

# def loss(p):
#         a_new, b_new = p
#         vol_left = quadratic(extreme_left, a_new, b_new, c_new)
#         vol_right = quadratic(extreme_right, a_new, b_new, c_new)
#         avg_extreme = (vol_left + vol_right) / 2
#         return (avg_extreme - target_extreme_vol) ** 2

def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

def generate_svi_initial_guess(k, w):
    a_init = np.min(w) * 0.9
    m_init = k[np.argmin(w)]
    
    slope_left = (w[1] - w[0]) / (k[1] - k[0])
    slope_right = (w[-1] - w[-2]) / (k[-1] - k[-2])
    b_init = max(abs(slope_left), abs(slope_right), 0.01)
    
    rho_init = (slope_right - slope_left) / (slope_right + slope_left + 1e-8)
    rho_init = np.clip(rho_init, -0.9, 0.9)
    
    sigma_init = np.std(k) * 0.5

    return [a_init, b_init, rho_init, m_init, sigma_init]


def fit_svi(strikes, ivs, spot, t):
    k = np.log(strikes / spot)
    total_variance = (ivs ** 2) * t

    def svi_loss(params):
        a, b, rho, m, sigma = params
        model_var = svi_total_variance(k, a, b, rho, m, sigma)
        return np.sum((model_var - total_variance) ** 2)

    # Initial guess and bounds
    # initial_guess = [0.01, 0.1, 0.0, 0.0, 0.1]
    initial_guess = generate_svi_initial_guess(k, total_variance)
    bounds = [(0, None), (0, None), (-0.999, 0.999), (None, None), (1e-5, None)]

    res = minimize(svi_loss, initial_guess, bounds=bounds, method='L-BFGS-B')
    return res.x if res.success else None



def svi_iv(K, F, t, params):
    k = np.log(K / F)
    a, b, rho, m, sigma = params
    w = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(w / t)

# --- Main Execution ---
input_file = 'ratio_iv_data.xlsx'
sheet_name = 'data_sheet'
output_file = 'ratio_iv_data_output.xlsx'
itm_req=input("PLease Save your Excel File(but do not close it)\nEnter ITM: ").upper()
# Ask user for mode: skew manipulation or direct IV cases
mode = input("Choose input mode:\n1. ATM/Far Strike Vol Change (skew manipulation)\n2. Direct IV Change cases (manual input)\nEnter 1 or 2: ").strip()

if mode == "1":
    atm_vol_change = float(input('Enter the change in ATM vol for the skew case (in %, default 1): ').strip() or "1") / 100   # 0.01
    far_strike_vol_change = float(input('Enter the change in Extreme/Far vol for the skew case (in %, default 3): ').strip() or "3") / 100   # 0.03
    input_iv_cases = None  # Will be generated later in the code as per original logic
elif mode == "2":
    atm_vol_change = 0.01
    far_strike_vol_change = 0.03
    while True:
        input_iv_cases = input("Enter 7 IV change cases separated by spaces (e.g. 0.1 0.08 0.04 0.02 0.06 0.1 0.14): ").strip()
        input_iv_cases = [parse_float(x) for x in input_iv_cases.split()]
        if len(input_iv_cases) == 7 and all(v is not None for v in input_iv_cases):
            break
        print("❌ Please enter exactly 7 valid float values separated by spaces.")
    if len(input_iv_cases) != 7:
        raise ValueError("You must enter exactly 7 IV values.")
else:
    raise ValueError("Invalid input. Please enter 1 or 2.")

# Check if output file is locked
if os.path.exists(output_file):
    try:
        os.remove(output_file)
    except PermissionError:
        print(f"❌ Cannot overwrite '{output_file}' – please close it in Excel and try again.")
        exit()

# Load original data


#copy_file = 'ratio_iv_data_copy.xlsx'

# try:
#     shutil.copy(input_file, copy_file)
#     df = pd.read_excel(copy_file, sheet_name=sheet_name) #hahah
#     print("asas")
#     print(df.head())  # Or your further analysis
# except FileNotFoundError:
#     print(f"Error: File '{input_file}' not found.")
# except Exception as e:
#     print(f"An error occurred: {e}")

df = pd.read_excel(input_file, sheet_name=sheet_name)
print("Column names found in Excel:")
print(df.columns.tolist())

# Group by expiry and type
grouped = df.groupby(['Expiry', 'Type'])

# Process each group
results = []

# Load and parse JSON
with open('output_27jun.json', 'r', encoding='utf-8') as f:
    raw = f.read()
    data = json.loads(raw)  # first parse
    if isinstance(data, str):
        data = json.loads(data)  # second parse if it's still a string

for (expiry, opt_type), group in grouped:
    group = group.copy()
    group = group.dropna(subset=['Spot', 'Strike', 'IV', 'DTE(% of year)'])
    group = group.copy()
    group['IV'] = pd.to_numeric(group['IV'], errors='coerce')
    group = group[(group['IV'] > 0)]


    if len(group) < 5:
        continue

    x = group['Strike'].values
    y = group['IV'].values
    y_m = group['IV_manual'].values
    S = group['Spot'].iloc[0]
    t = group['DTE(% of year)'].iloc[0]

    # Restrict strikes for fitting SVI
    # fit_group = group[group['Strike'] <= 115]
    # fit_x = fit_group['Strike'].values
    # fit_y = fit_group['IV'].values
    # fit_S = fit_group['Spot'].iloc[0] if not fit_group.empty else S
    # fit_t = fit_group['DTE(% of year)'].iloc[0] if not fit_group.empty else t

    # try:
    #     params = fit_svi(fit_x, fit_y, fit_S, fit_t)
    #     if params is None:
    #         print(f"SVI fit failed for ({expiry}, {opt_type})")
    #         continue
    #     a, b, rho, m, sigma = params

    # except Exception as e:
    #     print(f"Skipping group ({expiry}, {opt_type}) due to curve_fit error: {e}")
    #     continue
    try:
        params = fit_svi(x, y, S, t)
        params_m= fit_svi(x, y_m, S, t) 
        if params is None:
            print(f"SVI fit failed for ({expiry}, {opt_type})")
            continue
        a, b, rho, m, sigma = params
        a_m, b_m, rho_m, m_m, sigma_m = params_m

    except Exception as e:
        print(f"Skipping group ({expiry}, {opt_type}) due to curve_fit error: {e}")
        continue

    # Now use original data for applying synthetic IVs
    unfiltered_group = grouped.get_group((expiry, opt_type))

    atm_x = S
    strikes_sorted = sorted(x, key=lambda k: k - atm_x)
    extreme_left = strikes_sorted[0]
    extreme_right = strikes_sorted[-1]
    skew_params = list(params)
    atm_vol = svi_iv(S, S, t, params)
    target_atm_vol = atm_vol + atm_vol_change
    target_extreme_vol = atm_vol + far_strike_vol_change
    atm_w = (target_atm_vol ** 2) * t
    skew_params[0] += (atm_w - svi_total_variance(0, *params)) #bump a
    # For extra steep wings, increase b
    skew_params[1] *= (1 + far_strike_vol_change / atm_vol_change)

    

    unfiltered_group = unfiltered_group.reset_index(drop=True)

    for i, row in unfiltered_group.iterrows():
        # print(f"Processing row {i+1}/{len(unfiltered_group)}: Strike={row['Strike']}, IV={row['IV']}, Expiry={row['Expiry']}, Type={row['Type']}")
        if i>3 and i<len(unfiltered_group)-3:
            #print(unfiltered_group.iloc[i-3]['IV'], unfiltered_group.iloc[i-2]['IV'],unfiltered_group.iloc[i-1]['IV'],unfiltered_group.iloc[i]['IV'],unfiltered_group.iloc[i+1]['IV'],unfiltered_group.iloc[i+2]['IV'],unfiltered_group.iloc[i+3]['IV'])
            vol_cases=[unfiltered_group.iloc[i-3]['IV'], unfiltered_group.iloc[i-2]['IV'],unfiltered_group.iloc[i-1]['IV'],unfiltered_group.iloc[i]['IV'],unfiltered_group.iloc[i+1]['IV'],unfiltered_group.iloc[i+2]['IV'],unfiltered_group.iloc[i+3]['IV']]
            vol_cases_m=[unfiltered_group.iloc[i-3]['IV_manual'], unfiltered_group.iloc[i-2]['IV_manual'],unfiltered_group.iloc[i-1]['IV_manual'],unfiltered_group.iloc[i]['IV_manual'],unfiltered_group.iloc[i+1]['IV_manual'],unfiltered_group.iloc[i+2]['IV_manual'],unfiltered_group.iloc[i+3]['IV_manual']]
        else:
            vol_cases=[np.nan] * 7    
            vol_cases_m=[np.nan] * 7   
        K = row['Strike']
        orig_iv=row['IV']
        orig_iv_m=row['IV_manual']
        expiry=row['Expiry']
        if not pd.isna(row['Price_orig']):
            Price_orig = split_mixed_add(float(row['Price_orig'])) 
            # print("answer",row['Price_orig'],Price_orig, K, expiry) 
        else:
            None
        # print(orig_iv, K, expiry)
        synth_iv = orig_iv if not pd.isna(orig_iv) else svi_iv(K, S, t, params)
        synth_iv_m = orig_iv if not pd.isna(orig_iv_m) else svi_iv(K, S, t, params_m)
        
        # print(orig_iv, K, expiry)
        
        if mode == "1":
            skewed_iv = svi_iv(K, S, t, skew_params)
        else:
            # Use the center value (ATM) from the user-provided IV cases
            #skewed_iv = input_iv_cases[3]+synth_iv if input_iv_cases is not None else synth_iv
            skewed_iv = input_iv_cases[3]+synth_iv_m if input_iv_cases is not None else synth_iv_m
        net_position = 0.0
        avg_buy_price=0.0
        avg_sell_price=0.0  
        buy_qty=0.0
        sell_qty=0.0
        open_avg=0.0
        sod_price=0.0
        sod_qty=0.0
        sod_price_type=None
        enter_premium=0.0
        sod_net=0.0
        pos=0.0
        prem_change_since_sod_or_open=buy_qty*avg_buy_price-sell_qty*avg_sell_price
        # print(spot_cases)
        #synth_iv_cases=[svi_iv(K-spot_case, S, t, params) for spot_case in spot_cases]
        synth_iv_cases = [
            vol_case if not pd.isna(vol_case) else svi_iv(K - spot_case, S, t, params)
            for vol_case, spot_case in zip(vol_cases, spot_cases)
        ]
        synth_iv_cases_m = [
            vol_case if not pd.isna(vol_case) else svi_iv(K - spot_case, S, t, params_m)
            for vol_case, spot_case in zip(vol_cases, spot_cases)
        ]
        if mode == "1":
            skew_iv_cases = [svi_iv(K - spot_case, S, t, skew_params) for spot_case in spot_cases]
        else:
            # Add synth_iv_cases and input_iv_cases elementwise
            skew_iv_cases = [
            (synth + inp) if (synth is not None and inp is not None) else None
            for synth, inp in zip(synth_iv_cases_m, input_iv_cases)
            ]
            # skew_iv_cases = [
            # (synth + inp) if (synth is not None and inp is not None) else None
            # for synth, inp in zip(synth_iv_cases, input_iv_cases)
            # ]
        # print(synth_iv_cases)
        #iv_from_prem=iv_from_premium(S, K, Price_orig, orig_iv, t, opt_type, synth_iv_cases, spot_cases)
        synth_greeks = black_scholes_with_greeks(S, K, synth_iv, t, opt_type, synth_iv_cases, spot_cases)
        skewed_greeks = black_scholes_with_greeks(S, K,skewed_iv, t, opt_type, skew_iv_cases, spot_cases)

        # Calculate net position
        for entry in data:
            itm = entry['ITM']
            # print(f"ITM: {itm}")
            for position in entry['Positions']:
                if itm != itm_req:  
                    continue
                instrument = position['Instrument']
                net_position = position['NetPosition']

                
                # print(f"  Instrument: {instrument}, Net Position: {net_position}")
                line = (f"  Instrument: {instrument}, Net Position: {net_position}")

                # Step 1: Remove the comma
                line = line.replace(",", "")

                # Step 2: Split on whitespace
                parts = line.split()

                # Step 3: Manually split the 'C108' into 'C' and '108'
                # if parts[3].startswith("C") or parts[3].startswith("P"):
                if parts[3].startswith("C"):
                    Type = "Call"
                elif parts[3].startswith("P"):
                    Type = "Put"
                Strike = parts[3][1:]
                parts = parts[:3] + [Type, Strike] + parts[4:]
                Position= parts[7]

                # Result: 8 parts
                if parts[1]=="OZN":
                    expi="1|G|XCBT:O:"+"21:"+f"{convert_code_to_yyyymm(parts[2])}"+":"
                    # print(Type,Strike,expi, Position,opt_type,K,expiry,itm_req)
                elif parts[1][:3] not in month_map:
                    parts1=parts[1]
                    if parts1.startswith("ZN"):
                        parts1 = "TY" + parts1[2:]
                    expi="1|G|XCBT:O:"+parts1+":"+f"{convert_hyphen_code(parts[2][3:])}"+":"
                    print(Type,Strike,expi, Position,opt_type,K,expiry,itm_req)

                if Type == opt_type and float(Strike) == float(K) and expi == expiry and itm == itm_req:
                    pos=Position
                    # print(f"Position: {pos}, Expiry: {expi}")
                    avg_buy_price=position['AverageBuyPrice']*64
                    avg_sell_price=position['AverageSellPrice']*64    
                    buy_qty=position['BuyQuantity']
                    sell_qty=position['SellQuantity']
                    open_avg=position['OpenAveragePrice']
                    sod_price=position['SodPrice']*64
                    sod_qty=position['SodQuantity']
                    sod_price_type=position['SodPriceType']
                    enter_premium=avg_buy_price*buy_qty-avg_sell_price*sell_qty
                    sod_net=sod_price*sod_qty
                    print(Type,Strike,expiry, pos)
                    break
                


        result_row = {
            'Expiry': expiry,
            'Type': opt_type,
            'Strike': K,
            'Spot': S,
            'DTE(% of year)': t,
            'IV': orig_iv,
            # 'From_Prem_IV': iv_from_prem,
            'Synthetic_IV': synth_iv,
            'Skewed_IV': skewed_iv,
            'Positions': float(pos),
            'EnterPremium': enter_premium,
            'EnterBuyPrice': avg_buy_price,
            'EnterBuyQty': buy_qty,   
            'EnterSellPrice': avg_sell_price,
            'EnterSellQty': sell_qty,
            'SodNet': sod_net,
            'SodPrice': sod_price, 
            'SodQty': sod_qty,
            'SodPriceType': sod_price_type,
            'OpenAvgPrice': open_avg,
            'Price_orig': Price_orig,   
        }

        if synth_greeks:
            # Old snippet commented out
            # for k, v in synth_greeks.items():
            #     result_row[f'{k}'] = round(v, 4)
            # if skewed_greeks:
            #     for k, v in skewed_greeks.items():
            #         result_row[f'Skewed_{k}'] = round(v, 4)

            # New snippet
            for k, v in synth_greeks.items():
                if isinstance(v, list):  # Check if the value is a list
                    result_row[f'{k}'] = [round(item, 4) if item is not None else None for item in v]
                else:
                    result_row[f'{k}'] = round(v, 4)
            if skewed_greeks:
                for k, v in skewed_greeks.items():
                    if isinstance(v, list):  # Check if the value is a list
                        result_row[f'Skewed_{k}'] = [round(item, 4) if item is not None else None for item in v]
                    else:
                        result_row[f'Skewed_{k}'] = round(v, 4)

        results.append(result_row)
        pos=0.0

# Save results
output_df = pd.DataFrame(results)
import pandas as pd
# Example structure (replace with your actual DataFrame)
# df = pd.read_csv("your_data.csv")

# Columns to weight
original_columns = ['Premium', 'Delta', 'Gamma', 'Vega', 'Theta', 's_cases',"t_cases_curr","t_cases_1day","t_cases_3day","t_cases_7day","t_cases_10day","t_cases_14day"]
skewed_columns   = ['Skewed_Premium', 'Skewed_Delta', 'Skewed_Gamma', 'Skewed_Vega', 'Skewed_Theta' ,'Skewed_s_cases']
# print(df.columns.tolist())

# Weighted sum for original columns
weighted_sums = {}
# Filter and list string values in 'Positions' column
string_values = output_df[output_df['Positions'].apply(lambda x: isinstance(x, str))]['Positions'].tolist()

# Print them
for val in string_values:
    print(val)


for col in original_columns:
    if col=='Premium':
        weighted_sums[f"Weighted_{col}"] = (output_df[col] * output_df['Positions']).sum()
    elif col=='s_cases':
        # print(f"Dimensions of output_df: {len(output_df[col].iloc[0]) if isinstance(output_df[col].iloc[0], (list, tuple)) else 'N/A'}")
        # print(f"spot_cases: {output_df[col].tolist()}")
        weighted_sums[f"Weighted_{col}"] = [
            sum(output_df[col].iloc[i][j] * output_df['Positions'].iloc[i] 
            for i in range(len(output_df)) if isinstance(output_df[col].iloc[i], (list, tuple)))
            for j in range(len(spot_cases))
        ]
    elif col[0:8]=='t_cases_':
        # print(f"Dimensions of output_df: {len(output_df[col].iloc[0]) if isinstance(output_df[col].iloc[0], (list, tuple)) else 'N/A'}")
        # print(f"spot_cases: {output_df[col].tolist()}")
        weighted_sums[f"Weighted_{col}"] = [
            sum(output_df[col].iloc[i][j] * output_df['Positions'].iloc[i] 
            for i in range(len(output_df)) if isinstance(output_df[col].iloc[i], (list, tuple)))
            for j in range(len(spot_cases))
        ] 
    else:
        weighted_sums[f"Weighted_{col}"] = (output_df[col] * output_df['Positions']).sum()

# Weighted sum for skewed columns
for col in skewed_columns:
    if col == 'Skewed_s_cases':
        weighted_sums[f"Weighted_{col}"] = [
            sum(output_df[col].iloc[i][j] * output_df['Positions'].iloc[i] 
            for i in range(len(output_df)) if isinstance(output_df[col].iloc[i], (list, tuple)))
            for j in range(len(spot_cases))
        ]
    else:
        weighted_sums[f"Weighted_{col}"] = (output_df[col] * output_df['Positions']).sum()

net_calls = output_df[output_df['Type'] == 'Call']['Positions'].sum()
net_puts  = output_df[output_df['Type'] == 'Put']['Positions'].sum()
gross_calls = output_df[output_df['Type'] == 'Call']['Positions'].abs().sum()
gross_puts  = output_df[output_df['Type'] == 'Put']['Positions'].abs().sum()
sod_entry_pnl=weighted_sums['Weighted_Premium']-(output_df['SodNet']+output_df['EnterPremium']).sum()
sod_pnl=weighted_sums['Weighted_Premium']-output_df['SodNet'].sum()
skew_sod_entry_pnl=weighted_sums['Weighted_Skewed_Premium']-(output_df['SodNet']+output_df['EnterPremium']).sum()
skew_sod_pnl=weighted_sums['Weighted_Skewed_Premium']-output_df['SodNet'].sum()
Spot_cases=[spot_case+output_df['Spot'][1] for spot_case in spot_cases]
sod_entry_pnl_spot_cases=[spot_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_case in weighted_sums['Weighted_s_cases']]
sod_entry_pnl_spot_t_cases_curr=[spot_t_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_t_case in weighted_sums['Weighted_t_cases_curr']]
sod_entry_pnl_spot_t_cases_1day=[spot_t_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_t_case in weighted_sums['Weighted_t_cases_1day']]
sod_entry_pnl_spot_t_cases_3day=[spot_t_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_t_case in weighted_sums['Weighted_t_cases_3day']]
sod_entry_pnl_spot_t_cases_7day=[spot_t_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_t_case in weighted_sums['Weighted_t_cases_7day']]
sod_entry_pnl_spot_t_cases_10day=[spot_t_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_t_case in weighted_sums['Weighted_t_cases_10day']]
sod_entry_pnl_spot_t_cases_14day=[spot_t_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_t_case in weighted_sums['Weighted_t_cases_14day']]
sod_pnl_spot_cases=[spot_case-output_df['SodNet'].sum() for spot_case in weighted_sums['Weighted_s_cases']]
skew_sod_entry_pnl_spot_cases=[spot_case-(output_df['SodNet']+output_df['EnterPremium']).sum() for spot_case in weighted_sums['Weighted_Skewed_s_cases']]
skew_sod_pnl_spot_cases=[spot_case-output_df['SodNet'].sum() for spot_case in weighted_sums['Weighted_Skewed_s_cases']]

print(f"\nITM: {itm_req}\n")
print(f"Net Call Position: {net_calls}")
print(f"Net Put Position: {net_puts}\n")
print(f"Total Call Positions: {gross_calls}")
print(f"Total Put Positions:  {gross_puts}\n")
print(f"Approx PnL since SoD or same day entry(ticks): {sod_entry_pnl}")
print(f"Approx PnL since SoD(ticks):  {sod_pnl}\n")
print(f"Skew Case: ATM Vol Change {100*atm_vol_change}%, Extreme Vol Change {100*far_strike_vol_change}%")
print(f"Approx Skewed PnL since SoD or same day entry(ticks): {skew_sod_entry_pnl}")
print(f"Approx Skewed PnL since SoD(ticks):  {skew_sod_pnl}\n")
print("spot",Spot_cases)
print(f"\nApprox PnL since SoD or same day entry(ticks): {sod_entry_pnl_spot_cases}")
print(f"Approx PnL since SoD(ticks):  {sod_pnl_spot_cases}\n")
print(f"Skew Case: ATM Vol Change {100*atm_vol_change}%, Extreme Vol Change {100*far_strike_vol_change}%")
print(f"Approx Skewed PnL since SoD or same day entry(ticks): {skew_sod_entry_pnl_spot_cases}")
print(f"Approx Skewed PnL since SoD(ticks):  {skew_sod_pnl_spot_cases}\n")

# Display result
for key, value in weighted_sums.items():
    if not isinstance(value, list):  # Ignore spot_cases which are lists
        print(f"{key}: {value:.4f}")

output_df.to_excel(output_file, index=False)
print(f"✅ Finished! Output saved to: {output_file}")


plt.figure(figsize=(10, 6))

# Plot dotted original SOD PnL
plt.plot(Spot_cases, sod_entry_pnl_spot_cases, 'r--', label='PnL since SOD/Entry')
plt.plot(Spot_cases, sod_entry_pnl_spot_t_cases_1day, 'g--', label='PnL 1 Day Forward')
plt.plot(Spot_cases, sod_entry_pnl_spot_t_cases_3day, 'm--', label='PnL 3 Day Forward')
plt.plot(Spot_cases, sod_entry_pnl_spot_t_cases_7day, 'c--', label='PnL 7 Day Forward')
plt.plot(Spot_cases, sod_entry_pnl_spot_t_cases_10day, 'y--', label='PnL 10 Day Forward')
plt.plot(Spot_cases, sod_entry_pnl_spot_t_cases_14day, 'k--', label='PnL 14 Day Forward')
# Plot solid skew-adjusted SOD PnL
plt.plot(Spot_cases, skew_sod_entry_pnl_spot_cases, 'b-', label='Skewed PnL since SOD/Entry')

# Axes and grid
plt.xlabel('Spot Price')
plt.ylabel('PnL')
plt.title('PnL vs Spot Cases')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plot vol skew vs strikes for each expiry, call, and put
plt.figure(figsize=(12, 8))

# # Group by expiry and type again for plotting
for (expiry, opt_type), group in grouped:
    # group = group.copy()
    # group = group.dropna(subset=['Strike', 'IV'])
    # group = group[(group['IV'] > 0) & np.isfinite(group['IV'])]
    group = group.copy()
    group['IV'] = pd.to_numeric(group['IV'], errors='coerce')
    group = group[(group['IV'] > 0)]

    if len(group) < 5:
        continue

    strikes = group['Strike'].values
    ivs = group['IV'].values

    plt.plot(strikes, ivs, label=f'{expiry} - {opt_type}')

plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Volatility Skew vs Strike Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot predicted strike curve
plt.figure(figsize=(12, 8))

for (expiry, opt_type), group in grouped:
    # group = group.copy()
    # group = group.dropna(subset=['Strike', 'IV'])
    # group = group[(group['IV'] > 0) & np.isfinite(group['IV'])]
    group = group.copy()
    group['IV'] = pd.to_numeric(group['IV'], errors='coerce')
    group = group[(group['IV'] > 0)]

    if len(group) < 5:
        continue

    strikes = group['Strike'].values
    ivs = group['IV'].values

    S = group['Spot'].iloc[0]
    t = group['DTE(% of year)'].iloc[0]

    try:
        params = fit_svi(strikes, ivs, S, t)
        if params is None:
            # print(f"SVI fit failed for ({expiry}, {opt_type})")
            continue

        # Generate predicted IVs for a wider range of strikes
        extended_strikes = np.arange(106, 120, 0.5)  # Generate strikes from 106 to 120 with step 0.5
        predicted_ivs = [svi_iv(K, S, t, params) for K in extended_strikes]

        # Print comparison of original and predicted IVs for all strikes
        # print(f"\nComparison for ({expiry}, {opt_type}):")

        # for strike, pred_iv in zip(extended_strikes, predicted_ivs):
        #     print(f"Strike: {strike:.2f}, Predicted IV: {pred_iv:.4f}")

        # Plot original IVs
        plt.plot(strikes, ivs, 'o', label=f'Original {expiry} - {opt_type}')

        # Plot predicted IVs
        plt.plot(extended_strikes, predicted_ivs, '-', label=f'Predicted {expiry} - {opt_type}')
    except Exception as e:
        print(f"Skipping group ({expiry}, {opt_type}) due to error: {e}")
        continue

plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Original vs Predicted Volatility Skew')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Print comparison of predicted and original IVs for all expiries and option types
print("\nComparison of Predicted and Original IVs for All Expiries and Option Types:")
# for (expiry, opt_type), group in grouped:
#     # group = group.copy()
#     # group = group.dropna(subset=['Strike', 'IV'])
#     # group = group[(group['IV'] > 0) & np.isfinite(group['IV'])]
#     group = group.copy()
#     group['IV'] = pd.to_numeric(group['IV'], errors='coerce')
#     group = group[(group['IV'] > 0)]

#     # Restrict strikes to be below 116
#     group = group[group['Strike'] < 115]

#     if len(group) < 5:
#         print(f"Not enough data points for ({expiry}, {opt_type}).")
#         continue

#     strikes = group['Strike'].values
#     ivs = group['IV'].values

#     S = group['Spot'].iloc[0]
#     t = group['DTE(% of year)'].iloc[0]

#     try:
#         params = fit_svi(strikes, ivs, S, t)
#         if params is None:
#             print(f"SVI fit failed for ({expiry}, {opt_type})")
#             continue

#         # Generate predicted IVs for the original strikes
#         predicted_ivs = [svi_iv(K, S, t, params) for K in strikes]

#         # Print comparison of original and predicted IVs
#         print(f"\nComparison for ({expiry}, {opt_type}):")
#         for strike, orig_iv, pred_iv in zip(strikes, ivs, predicted_ivs):
#             print(f"Strike: {strike:.2f}, Original IV: {orig_iv:.4f}, Predicted IV: {pred_iv:.4f}")
#     except Exception as e:
#         print(f"Error processing ({expiry}, {opt_type}): {e}")

#[(114.75, 114.5, 114.25, 114, 113.75, 113.5, 113.25, 113, 112.75, 112.5, 112.25, 112, 111.75, 111.5, 111.25, 111, 110.75, 110.5, 110.25, 110, 109.75),
# (0.0996, 0.0972, 0.0937, 0.093, 0.0908, 0.0886, 0.0875, 0.0859, 0.0846, 0.0834, 0.0822, 0.0808, 0.079, 0.0795, 0.0786, 0.0779, 0.078, 0.0798, 0.0795, 0.0835, 0.0819), 
# (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None), 
# (0.0869, 0.0868, 0.0853, 0.0827, 0.082, 0.0798, 0.0786, 0.0781, 0.0768, 0.0757, 0.0756, 0.0734, 0.073, 0.0726, 0.0723, 0.0723, 0.0714, 0.0717, 0.0719, 0.0728, 0.0755), 
# (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None), 
# (0.0896, 0.0891, 0.0877, 0.0854, 0.0845, 0.0835, 0.0824, 0.081, 0.0802, 0.0797, 0.0787, 0.0778, 0.0772, 0.0768, 0.0768, 0.0762, 0.0762, 0.0758, 0.0768, 0.0763, 0.0783), (0.0829, None, 0.0813, None, 0.0792, None, 0.0771, 0.0763, 0.0756, 0.0747, 0.0745, 0.0739, 0.0737, 0.0734, 0.0732, 0.0735, 0.0726, 0.0729, 0.0732, 0.0736, 0.0744), (None, None, None, 0.0758, None, 0.0747, None, 0.0742, None, 0.0733, None, 0.0729, None, 0.0729, None, 0.0733, None, 0.0735, None, None, None), (None, None, None, 0.0758, None, 0.0747, None, 0.0742, None, 0.0733, None, 0.0729, None, 0.0729, None, 0.0733, None, 0.0735, None, None, None)]