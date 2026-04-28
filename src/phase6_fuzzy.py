# ============================================================
# PHASE 6 — Fuzzy Logic Translation Layer
# ============================================================
# GOAL: Convert raw PM2.5 numbers into human-readable
#       health advisories using fuzzy inference rules.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ── 1. Define the universe of discourse ──────────────────────
# This is just the range of possible values for each variable.
# np.arange(0, 501, 1) creates [0, 1, 2, ..., 500]

# Input: PM2.5 concentration in μg/m³
pm25_var = ctrl.Antecedent(np.arange(0, 501, 1), 'pm25')

# Output: Risk level 0–100 (0=safe, 100=emergency)
risk_var = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# ── 2. Define membership functions ───────────────────────────
# A membership function defines how strongly a value
# "belongs" to a category. Values are between 0 and 1.
#
# fuzz.trimf(universe, [a, b, c]):
#   a = where membership starts rising from 0
#   b = where membership peaks at 1
#   c = where membership falls back to 0
#
# These thresholds are based on:
#   - Chinese AQI standard (GB 3095-2012)
#   - WHO Air Quality Guidelines (2021)

pm25_var['good']           = fuzz.trimf(pm25_var.universe,
                                         [0,   0,   35])
pm25_var['moderate']       = fuzz.trimf(pm25_var.universe,
                                         [15,  50,  75])
pm25_var['unhealthy_sens'] = fuzz.trimf(pm25_var.universe,
                                         [55,  115, 150])
pm25_var['unhealthy']      = fuzz.trimf(pm25_var.universe,
                                         [115, 200, 250])
pm25_var['very_hazardous'] = fuzz.trimf(pm25_var.universe,
                                         [200, 350, 500])

risk_var['low']       = fuzz.trimf(risk_var.universe, [0,   0,   30])
risk_var['medium']    = fuzz.trimf(risk_var.universe, [20,  40,  60])
risk_var['high']      = fuzz.trimf(risk_var.universe, [50,  70,  85])
risk_var['critical']  = fuzz.trimf(risk_var.universe, [75,  90,  100])
risk_var['emergency'] = fuzz.trimf(risk_var.universe, [90,  100, 100])

# ── 3. Define fuzzy rules ─────────────────────────────────────
# These encode expert knowledge as IF-THEN statements.
# The fuzzy system uses these rules to infer risk from pm25.

rule1 = ctrl.Rule(pm25_var['good'],           risk_var['low'])
rule2 = ctrl.Rule(pm25_var['moderate'],       risk_var['medium'])
rule3 = ctrl.Rule(pm25_var['unhealthy_sens'], risk_var['high'])
rule4 = ctrl.Rule(pm25_var['unhealthy'],      risk_var['critical'])
rule5 = ctrl.Rule(pm25_var['very_hazardous'], risk_var['emergency'])

# ── 4. Build and compile the control system ───────────────────
air_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
air_sim  = ctrl.ControlSystemSimulation(air_ctrl)

print("Fuzzy control system built successfully.")

# ── 5. Visualize membership functions ────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PM2.5 input memberships
x = np.arange(0, 501, 1)
axes[0].plot(x, fuzz.trimf(x, [0,   0,   35]),  label='Good',                color='#00c853')
axes[0].plot(x, fuzz.trimf(x, [15,  50,  75]),  label='Moderate',            color='#ffd600')
axes[0].plot(x, fuzz.trimf(x, [55,  115, 150]), label='Unhealthy (sensitive)', color='#ff6d00')
axes[0].plot(x, fuzz.trimf(x, [115, 200, 250]), label='Unhealthy',            color='#d50000')
axes[0].plot(x, fuzz.trimf(x, [200, 350, 500]), label='Very Hazardous',       color='#6a1b9a')
axes[0].set_title('PM2.5 Membership Functions\n(How much does a value belong to each category?)')
axes[0].set_xlabel('PM2.5 (μg/m³)')
axes[0].set_ylabel('Membership degree (0–1)')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Risk output memberships
r = np.arange(0, 101, 1)
axes[1].plot(r, fuzz.trimf(r, [0,   0,   30]),  label='Low',       color='#00c853')
axes[1].plot(r, fuzz.trimf(r, [20,  40,  60]),  label='Medium',    color='#ffd600')
axes[1].plot(r, fuzz.trimf(r, [50,  70,  85]),  label='High',      color='#ff6d00')
axes[1].plot(r, fuzz.trimf(r, [75,  90,  100]), label='Critical',  color='#d50000')
axes[1].plot(r, fuzz.trimf(r, [90,  100, 100]), label='Emergency', color='#6a1b9a')
axes[1].set_title('Risk Level Membership Functions\n(Output of the fuzzy system)')
axes[1].set_xlabel('Risk level (0–100)')
axes[1].set_ylabel('Membership degree (0–1)')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('saved_models/fuzzy_memberships.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: saved_models/fuzzy_memberships.png")

# ── 6. The translation function ───────────────────────────────

def translate_pm25(pm25_value: float) -> dict:
    """
    Convert a PM2.5 reading into a structured health advisory.

    Uses the fuzzy inference system to compute a risk score,
    then maps to human-readable category, advice, and actions.

    Args:
        pm25_value: PM2.5 concentration in μg/m³

    Returns:
        dict with keys: pm25, risk_score, category,
                        advice, actions, mask, color
    """
    # Clamp to valid range [0, 500]
    pm25_value = float(np.clip(pm25_value, 0.0, 500.0))

    # Run fuzzy inference
    air_sim.input['pm25'] = pm25_value
    air_sim.compute()
    risk_score = air_sim.output['risk']

    # Map to advisory based on PM2.5 thresholds
    if pm25_value <= 35:
        return {
            'pm25':       round(pm25_value, 1),
            'risk_score': round(risk_score, 1),
            'category':   'Good',
            'color_code': '#00c853',
            'advice':     'Air quality is excellent. No health implications.',
            'actions':    [
                'No precautions needed',
                'Safe for outdoor exercise and activities'
            ],
            'mask': False
        }

    elif pm25_value <= 75:
        return {
            'pm25':       round(pm25_value, 1),
            'risk_score': round(risk_score, 1),
            'category':   'Moderate',
            'color_code': '#ffd600',
            'advice':     'Air quality is acceptable. Sensitive individuals should take note.',
            'actions':    [
                'Sensitive groups (asthma, elderly, children) limit long outdoor exertion',
                'General public can proceed normally'
            ],
            'mask': False
        }

    elif pm25_value <= 150:
        return {
            'pm25':       round(pm25_value, 1),
            'risk_score': round(risk_score, 1),
            'category':   'Unhealthy for Sensitive Groups',
            'color_code': '#ff6d00',
            'advice':     'Sensitive groups face health risks. General public may feel minor effects.',
            'actions':    [
                'Elderly, children, and respiratory patients: stay indoors',
                'Reduce outdoor physical activity',
                'Wear N95 mask if you must go outside',
                'Keep windows closed'
            ],
            'mask': True
        }

    elif pm25_value <= 250:
        return {
            'pm25':       round(pm25_value, 1),
            'risk_score': round(risk_score, 1),
            'category':   'Unhealthy',
            'color_code': '#d50000',
            'advice':     'Everyone may experience serious health effects.',
            'actions':    [
                'Avoid all strenuous outdoor activity',
                'N95 mask is mandatory outdoors',
                'Keep windows and doors sealed',
                'Run air purifier indoors if available',
                'Seek medical help if you experience breathing difficulty'
            ],
            'mask': True
        }

    else:
        return {
            'pm25':       round(pm25_value, 1),
            'risk_score': round(risk_score, 1),
            'category':   'Very Hazardous',
            'color_code': '#6a1b9a',
            'advice':     'Health emergency. Extremely dangerous air quality.',
            'actions':    [
                'STAY INDOORS — close all windows and doors',
                'Use an air purifier continuously',
                'N95 mask mandatory if any outdoor exposure is unavoidable',
                'Children and elderly must not go outside under any circumstances',
                'Seek medical attention immediately if symptoms appear'
            ],
            'mask': True
        }


# ── 7. Test the fuzzy layer with sample values ────────────────
print("\n=== Fuzzy Logic Translation Test ===\n")
test_values = [10, 40, 80, 130, 200, 320, 450]

for val in test_values:
    result = translate_pm25(val)
    print(f"PM2.5 = {val:>5} μg/m³  |  "
          f"Risk = {result['risk_score']:5.1f}  |  "
          f"Category: {result['category']}")
    print(f"  Advice : {result['advice']}")
    print(f"  Actions: {result['actions'][0]}")
    print()