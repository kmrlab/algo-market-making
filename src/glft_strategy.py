import math
import logging
import time
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class GLFTState:
    """GLFT strategy state"""
    price: float = 0.0
    position: float = 0.0
    variance: float = config.INITIAL_VARIANCE
    volatility: float = config.BASE_VOLATILITY
    shock_multiplier: float = 1.0
    a_adaptive: float = config.A_BASE
    k_adaptive: float = config.K_BASE
    last_price: float = 0.0
    cycle_count: int = 0
    
    # Temporal inventory management parameters
    position_entry_time: Optional[float] = None
    position_entry_price: Optional[float] = None  # entry price for P&L calculation
    last_position_sign: int = 0  # -1, 0, 1 for tracking position direction changes
    
    # Adaptive update frequency
    smoothed_frequency: float = config.BASE_UPDATE_FREQUENCY
    
    # Z-score volume adaptation system
    current_z_score: float = 0.0
    last_volume_adjustment_info: Optional[Dict] = None


class GLFTStrategy:
    """
    GLFT market-making strategy with adaptive volatility
    """
    
    def __init__(self):
        self.state = GLFTState()
        self.logger = self._setup_logger()
        self.price_history = []
        
    def _setup_logger(self) -> logging.Logger:
        """Strategy logger setup"""
        logger = logging.getLogger('glft_strategy')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | STRATEGY | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)  # Show everything for strategy
        return logger
    
    def _get_z_score_parameters(self) -> Optional[Dict]:
        """Get all Z-score parameters based on single intensity setting"""
        if not config.Z_SCORE_VOLUME_ADAPTATION_ENABLED:
            return None
        
        intensity = config.Z_SCORE_INTENSITY
        
        return {
            'boost_factor': intensity,
            'reduce_factor': intensity * 0.6,  # Always less than boost
            'max_z_score': 2.5,
            'min_multiplier': max(0.6, 1.0 - intensity * 1.2),  # Adaptive
            'max_multiplier': min(2.0, 1.0 + intensity * 2.4),  # Adaptive
            'activation_threshold': 0.3,
            'conflict_reduction': 0.4,
            'neutral_factor': 0.7,
            'large_position_threshold': 0.6,
            'volatility_dampening': True
        }
    
    def update_volatility(self, current_price: float) -> None:
        """
        Update adaptive volatility through simplified EWMA with shock multiplier
        """
        if self.state.last_price > 0:
            # Calculate return
            current_return = math.log(current_price / self.state.last_price)
            # Removed detailed logging of each return
            
            # Update EWMA variance
            old_variance = self.state.variance
            self.state.variance = (config.LAMBDA * self.state.variance + 
                                 (1 - config.LAMBDA) * current_return**2)
            
            # Base volatility
            base_volatility = math.sqrt(self.state.variance * config.ANNUALIZATION_FACTOR)
            
            # Shock detector
            old_shock = self.state.shock_multiplier
            if self.state.variance > 0:
                shock_intensity = abs(current_return) / math.sqrt(self.state.variance)
                if shock_intensity > config.SHOCK_THRESHOLD:
                    self.state.shock_multiplier = min(
                        config.MAX_VOLATILITY_MULTIPLIER,
                        1 + (shock_intensity - config.SHOCK_THRESHOLD) * 0.5
                    )
                    self.logger.info(f"SHOCK DETECTED: Intensity={shock_intensity:.3f}, Multiplier={self.state.shock_multiplier:.3f}")
                else:
                    self.state.shock_multiplier = 1.0
            
            # Final volatility
            old_volatility = self.state.volatility
            self.state.volatility = base_volatility * self.state.shock_multiplier
            
            # Log only significant volatility changes
            if abs(self.state.volatility - old_volatility) > 0.001:
                self.logger.debug(f"Volatility: {old_volatility:.4f} -> {self.state.volatility:.4f}")
            
        self.state.last_price = current_price
    
    def update_position_timing(self, current_position: float) -> None:
        """
        Track position holding time for temporal decay
        """
        if not config.INVENTORY_TIME_DECAY_ENABLED:
            return
            
        current_sign = 0 if abs(current_position) < 0.01 else (1 if current_position > 0 else -1)
        
        # Check position state change
        if current_sign != self.state.last_position_sign:
            if current_sign == 0:
                # Position closed
                self.state.position_entry_time = None
                self.state.position_entry_price = None
                self.logger.debug("Position closed, timer and entry price reset")
            else:
                # New position or direction change
                self.state.position_entry_time = time.time()
                self.state.position_entry_price = self.state.price
                direction = "LONG" if current_sign > 0 else "SHORT"
                self.logger.info(f"New {direction} position detected, entry price: {self.state.price}")
            
            self.state.last_position_sign = current_sign
    
    def get_time_decay_multiplier(self) -> float:
        """
        Calculate temporal multiplier with loss enhancement (THOUSAND_CUTS protection)
        """
        if (not config.INVENTORY_TIME_DECAY_ENABLED or 
            self.state.position_entry_time is None or 
            abs(self.state.position) < 0.01):
            return 1.0
        
        # Base temporal decay
        time_in_position = time.time() - self.state.position_entry_time
        decay_progress = min(1.0, time_in_position / (config.INVENTORY_DECAY_TIME_MINUTES * 60))
        base_multiplier = 1.0 + decay_progress * config.INVENTORY_DECAY_MULTIPLIER
        
        # Loss enhancement (THOUSAND_CUTS protection)
        if config.LOSS_ENHANCEMENT_ENABLED and self.state.position_entry_price:
            unrealized_pnl = (self.state.price - self.state.position_entry_price) * self.state.position
            
            if unrealized_pnl < 0:  # Only on losses
                position_value = abs(self.state.position * self.state.position_entry_price)
                loss_ratio = abs(unrealized_pnl) / position_value
                
                # Additional multiplier: from 1.0 to LOSS_ENHANCEMENT_CAP on losses up to LOSS_THRESHOLD_MAX_ENHANCEMENT
                loss_multiplier = 1.0 + min(config.LOSS_ENHANCEMENT_CAP - 1.0, loss_ratio / config.LOSS_THRESHOLD_MAX_ENHANCEMENT)
                
                # Log significant enhancements
                if loss_multiplier > 1.1:
                    self.logger.info(f"THOUSAND_CUTS PROTECTION: Loss {loss_ratio:.2%}, "
                                   f"multiplier enhanced {base_multiplier:.2f} → {base_multiplier * loss_multiplier:.2f}")
                
                return base_multiplier * loss_multiplier
        
        return base_multiplier

    def adapt_parameters(self) -> None:
        """
        Adaptation of A and K parameters based on volatility
        """
        # Normalize volatility relative to base level
        vol_ratio = self.state.volatility / config.BASE_VOLATILITY
        
        # Unified adaptive coefficient with constraints
        volatility_adjustment = min(
            config.MAX_VOLATILITY_MULTIPLIER,
            max(1 / config.MAX_VOLATILITY_MULTIPLIER,
                vol_ratio ** config.VOLATILITY_SENSITIVITY)
        )
        
        # Parameter adaptation
        old_a = self.state.a_adaptive
        old_k = self.state.k_adaptive
        
        # High volatility: decrease A (fewer orders), increase K (more penalty)
        self.state.a_adaptive = config.A_BASE / volatility_adjustment
        self.state.k_adaptive = config.K_BASE * volatility_adjustment
        
        # Detailed volatility adaptation logging for debugging
        if abs(vol_ratio - 1.0) > 0.1:  # Log only on significant deviations
            self.logger.debug(f"VOLATILITY ADAPTATION: vol_ratio={vol_ratio:.3f}, "
                            f"adjustment={volatility_adjustment:.3f}, "
                            f"A: {old_a:.3f}→{self.state.a_adaptive:.3f}, "
                            f"K: {old_k:.2f}→{self.state.k_adaptive:.2f}")
    
    def calculate_z_score(self, current_price: float) -> float:
        """
        Calculate directional Z-score based on existing shock system
        """
        params = self._get_z_score_parameters()
        if not params or self.state.last_price <= 0:
            return 0.0
        
        # Use existing return calculation logic
        current_return = math.log(current_price / self.state.last_price)
        
        # Protect from division by zero
        if self.state.variance <= 0:
            return 0.0
        
        # Calculate directional Z-score
        raw_z_score = current_return / math.sqrt(self.state.variance)
        
        # Apply constraints
        clamped_z_score = max(-params['max_z_score'], min(params['max_z_score'], raw_z_score))
        
        # Adaptive sensitivity at high volatility
        if params['volatility_dampening']:
            volatility_dampening = min(1.0, config.BASE_VOLATILITY / self.state.volatility)
            clamped_z_score *= volatility_dampening
        
        self.state.current_z_score = clamped_z_score
        
        # Log significant deviations
        if abs(clamped_z_score) > 1.5:
            direction = "UP" if clamped_z_score > 0 else "DOWN"
            self.logger.info(f"SIGNIFICANT Z-SCORE: {clamped_z_score:.3f} ({direction}) - "
                            f"return={current_return:.4f}, vol={math.sqrt(self.state.variance):.4f}")
        
        return clamped_z_score
    
    def apply_z_score_volume_adaptation(self, base_bid_volume: float, base_ask_volume: float, 
                                       position_ratio: float) -> Dict:
        """
        Apply Z-score adaptations ONLY to volumes with signal conflict protection
        """
        params = self._get_z_score_parameters()
        if (not params or 
            abs(self.state.current_z_score) < params['activation_threshold']):
            return {
                'bid_volume': base_bid_volume,
                'ask_volume': base_ask_volume,
                'z_score': self.state.current_z_score,
                'adjustment_applied': False,
                'conflict_detected': False
            }
        
        z_score = self.state.current_z_score
        
        # Define signals for conflict checking
        inventory_signal = np.sign(position_ratio)  # -1, 0, +1
        z_score_signal = np.sign(z_score)          # -1, 0, +1
        
        # Check signal conflict
        signal_alignment = inventory_signal * z_score_signal
        large_position = abs(position_ratio) > params['large_position_threshold']
        
        # Determine correction strength
        if large_position:
            # Large position priority inventory management
            correction_strength = params['conflict_reduction'] * 0.5
            mode = "LARGE_POSITION_PRIORITY"
        elif signal_alignment > 0:
            # Signals aligned - full strength
            correction_strength = 1.0
            mode = "SIGNALS_ALIGNED"
        elif signal_alignment < 0:
            # Signals conflict - reduce strength
            correction_strength = params['conflict_reduction']
            mode = "SIGNALS_CONFLICTED"
        else:
            # One signal neutral - moderate strength
            correction_strength = params['neutral_factor']
            mode = "NEUTRAL_SIGNAL"
        
        # Calculate volume multipliers
        if z_score > 0:  # Price went up - stimulate selling
            ask_multiplier = 1 + abs(z_score) * params['boost_factor'] * correction_strength
            bid_multiplier = max(params['min_multiplier'], 
                               1 - abs(z_score) * params['reduce_factor'] * correction_strength)
            direction = "PRICE_UP"
        else:  # Price went down - stimulate buying
            bid_multiplier = 1 + abs(z_score) * params['boost_factor'] * correction_strength
            ask_multiplier = max(params['min_multiplier'],
                               1 - abs(z_score) * params['reduce_factor'] * correction_strength)
            direction = "PRICE_DOWN"
        
        # Apply safety constraints
        ask_multiplier = max(params['min_multiplier'], 
                            min(params['max_multiplier'], ask_multiplier))
        bid_multiplier = max(params['min_multiplier'],
                            min(params['max_multiplier'], bid_multiplier))
        
        # Calculate final volumes
        final_bid_volume = base_bid_volume * bid_multiplier
        final_ask_volume = base_ask_volume * ask_multiplier
        
        # Save adjustment information
        adjustment_info = {
            'z_score': z_score,
            'mode': mode,
            'direction': direction,
            'correction_strength': correction_strength,
            'bid_multiplier': bid_multiplier,
            'ask_multiplier': ask_multiplier,
            'inventory_signal': inventory_signal,
            'z_score_signal': z_score_signal,
            'signal_alignment': signal_alignment
        }
        
        self.state.last_volume_adjustment_info = adjustment_info
        
        # Logging
        if abs(z_score) > 1.0 or mode in ["SIGNALS_CONFLICTED", "LARGE_POSITION_PRIORITY"]:
            self.logger.debug(f"Z-SCORE VOLUME ADAPTATION: {mode} | Z={z_score:.2f} | "
                             f"Strength={correction_strength:.2f} | "
                             f"BID×{bid_multiplier:.2f}, ASK×{ask_multiplier:.2f}")
            
            if mode == "SIGNALS_CONFLICTED":
                self.logger.info(f"SIGNAL CONFLICT: Inventory={inventory_signal}, Z-score={z_score_signal}, "
                               f"Reduced strength to {correction_strength:.2f}")
        
        return {
            'bid_volume': final_bid_volume,
            'ask_volume': final_ask_volume,
            'z_score': z_score,
            'adjustment_applied': True,
            'conflict_detected': signal_alignment < 0,
            'adjustment_info': adjustment_info
        }
    
    def calculate_glft_components(self) -> Tuple[float, float]:
        """
        Calculate canonical GLFT model components
        Returns: (c1, c2)
        """
        # c1 component
        c1 = (1 / (config.XI * config.DELTA)) * math.log(
            1 + (config.XI * config.DELTA) / self.state.k_adaptive
        )
        
        # c2 component
        power_term = (self.state.k_adaptive / (config.XI * config.DELTA)) + 0.5
        base_term = 1 + (config.XI * config.DELTA) / self.state.k_adaptive
        
        c2 = (math.sqrt(config.GAMMA / (2 * self.state.a_adaptive * config.DELTA * self.state.k_adaptive)) *
              (base_term ** power_term))
        
        return c1, c2
    
    def calculate_quotes(self, fair_price: float) -> Dict[str, float]:
        """
        Calculate quotes with inventory management and Z-score adaptation ONLY for volumes
        """
        # 1. Calculate Z-score (NEW)
        z_score = self.calculate_z_score(fair_price)
        
        # 2. Base GLFT components (EXISTING - NO CHANGES)
        c1, c2 = self.calculate_glft_components()
        base_half_spread = c1 + (config.DELTA / 2) * self.state.volatility * c2
        base_skew = self.state.volatility * c2
        
        # 3. Inventory management (EXISTING - NO CHANGES)
        position_ratio = self.state.position / config.MAX_POSITION
        
        # 3.1. Spread expansion on large positions (NO CHANGES)
        spread_expansion = abs(position_ratio) * config.POSITION_SPREAD_EXPANSION
        final_half_spread = base_half_spread * (1 + spread_expansion)
        
        # 3.2. Asymmetric adjustment to stimulate position closing (NO CHANGES)
        base_inventory_skew = position_ratio * config.INVENTORY_FACTOR * base_skew
        time_multiplier = self.get_time_decay_multiplier()
        inventory_skew = base_inventory_skew * time_multiplier
        
        # 4. Base volumes (EXISTING)
        base_bid_volume, base_ask_volume = self.calculate_volumes(position_ratio)
        
        # 5. Apply Z-score adaptation ONLY to volumes (NEW)
        volume_adaptations = self.apply_z_score_volume_adaptation(
            base_bid_volume, base_ask_volume, position_ratio
        )
        
        # 6. Final quotes (EXISTING - NO CHANGES)
        bid_price = fair_price - final_half_spread - inventory_skew
        ask_price = fair_price + final_half_spread - inventory_skew
        
        # 7. Final volumes with constraints (MODIFIED)
        final_bid_volume = max(config.MIN_ORDER_SIZE, 
                              min(config.MAX_ORDER_SIZE, volume_adaptations['bid_volume']))
        final_ask_volume = max(config.MIN_ORDER_SIZE,
                              min(config.MAX_ORDER_SIZE, volume_adaptations['ask_volume']))
        
        return {
            'bid_price': round(bid_price, 2),
            'ask_price': round(ask_price, 2),
            'bid_volume': final_bid_volume,
            'ask_volume': final_ask_volume,
            'half_spread': final_half_spread,
            'skew': inventory_skew,
            'position_ratio': position_ratio,
            # New fields for Z-score volume adaptation monitoring
            'z_score': z_score,
            'volume_adaptations': volume_adaptations,
            'z_adjustment_applied': volume_adaptations['adjustment_applied'],
            'z_conflict_detected': volume_adaptations['conflict_detected']
        }
    
    def calculate_volumes(self, position_ratio: float) -> Tuple[float, float]:
        """
        Calculate asymmetric volumes based on position
        """
        if self.state.position > 0:  # Long position - stimulate selling
            ask_volume = config.BASE_ORDER_SIZE * (1 + abs(position_ratio) * config.VOLUME_ASYMMETRY_FACTOR)
            bid_volume = config.BASE_ORDER_SIZE * (1 - abs(position_ratio) * config.VOLUME_ASYMMETRY_FACTOR)
            volume_logic = "Long position: boosting ASK, reducing BID"
        elif self.state.position < 0:  # Short position - stimulate buying
            bid_volume = config.BASE_ORDER_SIZE * (1 + abs(position_ratio) * config.VOLUME_ASYMMETRY_FACTOR)
            ask_volume = config.BASE_ORDER_SIZE * (1 - abs(position_ratio) * config.VOLUME_ASYMMETRY_FACTOR)
            volume_logic = "Short position: boosting BID, reducing ASK"
        else:  # No position
            ask_volume = bid_volume = config.BASE_ORDER_SIZE
            volume_logic = "Neutral position: equal volumes"
        
        # Apply constraints
        final_bid_volume = max(config.MIN_ORDER_SIZE, min(config.MAX_ORDER_SIZE, bid_volume))
        final_ask_volume = max(config.MIN_ORDER_SIZE, min(config.MAX_ORDER_SIZE, ask_volume))
        
        # Removed detailed volume logging
        
        return final_bid_volume, final_ask_volume
    
    def update_state(self, current_price: float, current_position: float) -> Dict[str, float]:
        """
        Main method for updating strategy state
        """
        # Update state
        self.state.price = current_price
        self.state.position = current_position
        self.state.cycle_count += 1
        
        # 1. Update volatility
        self.update_volatility(current_price)
        
        # 2. Position time tracking (for temporal decay)
        self.update_position_timing(current_position)
        
        # 3. Parameter adaptation
        self.adapt_parameters()
        
        # 4. Calculate quotes
        quotes = self.calculate_quotes(current_price)
        
        # 4. Removed detailed logging - main logging now in main.py
        
        return quotes
    
    def get_strategy_info(self) -> Dict:
        """Get information about strategy state"""
        info = {
            'price': self.state.price,
            'position': self.state.position,
            'volatility': self.state.volatility,
            'shock_multiplier': self.state.shock_multiplier,
            'a_adaptive': self.state.a_adaptive,
            'k_adaptive': self.state.k_adaptive,
            'cycle_count': self.state.cycle_count
        }
        
        # Add temporal decay information
        if config.INVENTORY_TIME_DECAY_ENABLED:
            info['time_decay_multiplier'] = self.get_time_decay_multiplier()
            if self.state.position_entry_time:
                info['time_in_position_seconds'] = time.time() - self.state.position_entry_time
        
        return info
    
    def calculate_adaptive_frequency(self, performance_metrics: Optional[Dict] = None) -> float:
        """
        Calculate adaptive update frequency based on strategy state
        LOGIC: High volatility → SMALLER interval (more frequent updates)
        """
        # 1. Volatility factor (higher means more frequent updates needed)
        vol_ratio = self.state.volatility / config.BASE_VOLATILITY
        f_volatility = vol_ratio ** config.FREQUENCY_VOLATILITY_SENSITIVITY
        
        # 2. Position factor (large position → more frequent updates)
        position_ratio = abs(self.state.position) / config.MAX_POSITION
        f_position = 1.0 + position_ratio * config.FREQUENCY_POSITION_SENSITIVITY
        
        # 3. Shock factor (shocks → very frequent updates)
        f_shock = self.state.shock_multiplier ** config.FREQUENCY_SHOCK_SENSITIVITY
        
        # 4. Performance factor
        f_performance = 1.0
        if performance_metrics and 'avg_cycle_time_ms' in performance_metrics:
            target_time = config.BASE_UPDATE_FREQUENCY * 1000 * 0.3  # 30% of interval
            actual_time = performance_metrics['avg_cycle_time_ms']
            if actual_time > target_time:
                # If cycles are slow, increase interval (less frequent updates)
                f_performance = actual_time / target_time
        
        # 5. Overall activity multiplier
        activity_multiplier = f_volatility * f_position * f_shock
        
        # 6. INVERSE dependency: high activity → smaller interval
        raw_frequency = config.BASE_UPDATE_FREQUENCY / activity_multiplier
        
        # 7. Performance correction (if system is slow)
        raw_frequency *= f_performance
        
        # 8. Apply constraints
        constrained_frequency = max(config.MIN_UPDATE_FREQUENCY,
                                   min(config.MAX_UPDATE_FREQUENCY, raw_frequency))
        
        # 9. EWMA smoothing
        if not hasattr(self.state, 'smoothed_frequency'):
            self.state.smoothed_frequency = constrained_frequency
        else:
            self.state.smoothed_frequency = (
                config.FREQUENCY_SMOOTHING_LAMBDA * self.state.smoothed_frequency +
                (1 - config.FREQUENCY_SMOOTHING_LAMBDA) * constrained_frequency
            )
        
        # Log significant changes
        if abs(self.state.smoothed_frequency - config.BASE_UPDATE_FREQUENCY) > 1.0:
            activity_level = "HIGH" if activity_multiplier > 1.5 else "LOW" if activity_multiplier < 0.7 else "NORMAL"
            self.logger.info(f"ADAPTIVE FREQUENCY: {self.state.smoothed_frequency:.2f}s "
                            f"[{activity_level}] (vol:{f_volatility:.2f}, pos:{f_position:.2f}, "
                            f"shock:{f_shock:.2f}, activity:{activity_multiplier:.2f})")
        
        return self.state.smoothed_frequency

    def get_detailed_metrics(self) -> Dict:
        """Get detailed metrics for logging"""
        # Calculate GLFT components
        c1, c2 = self.calculate_glft_components()
        
        # Calculate base spread and expansion
        base_half_spread = c1 + (config.DELTA / 2) * self.state.volatility * c2
        position_ratio = self.state.position / config.MAX_POSITION
        spread_expansion = abs(position_ratio) * config.POSITION_SPREAD_EXPANSION
        
        # Final half spread with expansion
        final_half_spread = base_half_spread * (1 + spread_expansion)
        
        # Additional calculations
        vol_ratio = self.state.volatility / config.BASE_VOLATILITY
        
        # Analysis of volatility impact on spread
        volatility_component = (config.DELTA / 2) * self.state.volatility * c2
        c1_component = c1
        
        base_metrics = {
            'c1': c1,
            'c2': c2,
            'base_spread': base_half_spread * 2,  # Full base spread
            'final_spread': final_half_spread * 2,  # Full final spread
            'spread_expansion': spread_expansion,
            'spread_expansion_amount': base_half_spread * spread_expansion * 2,  # Absolute expansion
            'vol_ratio': vol_ratio,
            'position_ratio': position_ratio,
            'volatility_component': volatility_component * 2,  # Volatility contribution to spread
            'c1_component': c1_component * 2,  # c1 contribution to spread
            'volatility_adjustment': min(config.MAX_VOLATILITY_MULTIPLIER, 
                                       max(1 / config.MAX_VOLATILITY_MULTIPLIER,
                                           vol_ratio ** config.VOLATILITY_SENSITIVITY))
        }
        
        # Add Z-score metrics ONLY for volumes
        if config.Z_SCORE_VOLUME_ADAPTATION_ENABLED:
            params = self._get_z_score_parameters()
            base_metrics.update({
                'z_score': self.state.current_z_score,
                'z_score_abs': abs(self.state.current_z_score),
                'z_score_category': self._get_z_score_category(),
                'z_volume_adaptations_active': abs(self.state.current_z_score) > params['activation_threshold'] if params else False,
                'z_score_intensity': config.Z_SCORE_INTENSITY
            })
            
            # Information about last volume adjustment
            if self.state.last_volume_adjustment_info:
                adj_info = self.state.last_volume_adjustment_info
                base_metrics.update({
                    'z_adjustment_mode': adj_info['mode'],
                    'z_correction_strength': adj_info['correction_strength'],
                    'z_bid_multiplier': adj_info['bid_multiplier'],
                    'z_ask_multiplier': adj_info['ask_multiplier'],
                    'z_signal_alignment': adj_info['signal_alignment'],
                    'z_direction': adj_info['direction']
                })
        
        return base_metrics
    
    def _get_z_score_category(self) -> str:
        """Z-score categorization for logging"""
        z = abs(self.state.current_z_score)
        if z < 0.5:
            return "NORMAL"
        elif z < 1.0:
            return "MODERATE"
        elif z < 1.5:
            return "HIGH"
        elif z < 2.0:
            return "VERY_HIGH"
        else:
            return "EXTREME"