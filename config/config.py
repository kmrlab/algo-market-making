# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           1. GLOBAL SETTINGS                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# 1. GLOBAL SETTINGS  
# ┌─ Operating Mode ────────────────────────────────────────────────────────┐
PRODUCTION_MODE = False                     # Production mode (minimal logging)

# ┌─ API Connection ───────────────────────────────────────────────────────┐
API_KEY = ""              # Bybit API key
API_SECRET = ""  # Bybit API secret
TESTNET = False                             # Use testnet

# ┌─ Core Timings ─────────────────────────────────────────────────────────┐
UPDATE_FREQUENCY = 8                        # Base quote update frequency (sec)
TIMESTAMP_TOLERANCE_MS = 5000               # Allowed time deviation (ms)

# ┌─ Logging ──────────────────────────────────────────────────────────────┐
STRATEGY_LOG_FILE = "logs/glft.log"         # Main log file

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          2. TRADING PARAMETERS                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ┌─ Trading Instrument ──────────────────────────────────────────────────┐
SYMBOL = "SOLUSDT"                          # Trading pair
CATEGORY = "linear"                         # Contract type (perpetual futures)

# ┌─ Contract Parameters ──────────────────────────────────────────────────┐
TICK_SIZE = 0.01                           # Minimum price step
SIZE_PRECISION = 1                         # Size precision (decimal places)

# ┌─ Order Sizes ──────────────────────────────────────────────────────────┐
BASE_ORDER_SIZE = 1.0                      # Base order size (SOL)
MIN_ORDER_SIZE = 1.0                       # Minimum order size
MAX_ORDER_SIZE = 3.0                      # Maximum order size
MAX_POSITION = 10.0                        # Maximum position (SOL)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            3. GLFT MODEL                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ┌─ Core Guéant–Lehalle–Fernandez-Tapia Model Parameters ─────────────────┐

A_BASE = 0.15                              # Order intensity
K_BASE = 11                              # Inventory sensitivity
GAMMA = 1.5                                # Risk aversion coefficient (spread width)

# ┌─ Internal GLFT Parameters ──────────────────────────────────────────────┐
XI = 3                                   # Inventory penalty
DELTA = 1.0                                # Time step (normalizing coefficient)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            4. VOLATILITY                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ┌─ EWMA Volatility Model ─────────────────────────────────────────────────┐
LAMBDA = 0.8                               # EWMA smoothing parameter
ANNUALIZATION_FACTOR = 365*24*60/8         # Annualization coefficient
INITIAL_VARIANCE = 0.0005**2               # Initial variance
BASE_VOLATILITY = 0.05                    # Base volatility level

# ┌─ Volatility Adaptation ─────────────────────────────────────────────────┐
VOLATILITY_SENSITIVITY = 0.4               # Spread sensitivity to volatility
SHOCK_THRESHOLD = 0.8                      # Volatility shock detection threshold
MAX_VOLATILITY_MULTIPLIER = 2.5              # Maximum spread expansion during shocks

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         5. RISK MANAGEMENT                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ┌─ Inventory Management ──────────────────────────────────────────────────────┐
INVENTORY_FACTOR = 1.3                     # Asymmetric quote strength for position closing
POSITION_SPREAD_EXPANSION = 0.5            # Spread expansion for large positions
VOLUME_ASYMMETRY_FACTOR = 0              # Volume asymmetry for position closing

# ┌─ Position Time Decay ───────────────────────────────────────────────────────┐
INVENTORY_TIME_DECAY_ENABLED = True        # Enable time-based position closing enhancement
INVENTORY_DECAY_TIME_MINUTES = 20          # Full decay time (minutes)
INVENTORY_DECAY_MULTIPLIER = 3             # Maximum enhancement multiplier

# ┌─ Gradual Loss Protection "Thousand Cuts" ──────────────────────────────────────┐
LOSS_ENHANCEMENT_ENABLED = True            # Enable gradual loss protection ("Thousand Cuts")
LOSS_THRESHOLD_MAX_ENHANCEMENT = 0.005     # Loss threshold for maximum enhancement (0.5%)
LOSS_ENHANCEMENT_CAP = 2                 # Maximum enhancement multiplier



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        6. FREQUENCY SYSTEM                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ┌─ Adaptive Frequency Range ───────────────────────────────────────────────────┐
BASE_UPDATE_FREQUENCY = 8.0                # Base update frequency (sec)
MIN_UPDATE_FREQUENCY = 4.0                 # Minimum frequency (maximum speed)
MAX_UPDATE_FREQUENCY = 20.0               # Maximum frequency (minimum speed)

# ┌─ Frequency Influence Factors ──────────────────────────────────────────────────┐
FREQUENCY_VOLATILITY_SENSITIVITY = 0.6     # Sensitivity to volatility
FREQUENCY_POSITION_SENSITIVITY = 0.3       # Sensitivity to position size
FREQUENCY_SHOCK_SENSITIVITY = 0.8          # Sensitivity to volatility shocks
FREQUENCY_PERFORMANCE_WEIGHT = 0.3         # Performance factor weight

# ┌─ Frequency Smoothing ─────────────────────────────────────────────────────────┐
FREQUENCY_SMOOTHING_LAMBDA = 0.7           # EWMA coefficient for smoothing
FREQUENCY_UPDATE_THRESHOLD = 0.1           # Minimum change for update (sec)



# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                       7. Z-SCORE VOLUMES                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ┌─ Smart Volume Adaptation System ──────────────────────────────────────────────────┐
Z_SCORE_VOLUME_ADAPTATION_ENABLED = False   # Enable Z-score volume adaptation
Z_SCORE_INTENSITY = 0.6                   # Correction intensity (0.1-0.4)
                                           # 0.15=conservative, 0.25=balanced, 0.35=aggressive
