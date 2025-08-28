"""
██╗  ██╗███╗   ███╗██████╗ ██╗      █████╗ ██████╗ 
██║ ██╔╝████╗ ████║██╔══██╗██║     ██╔══██╗██╔══██╗
█████╔╝ ██╔████╔██║██████╔╝██║     ███████║██████╔╝
██╔═██╗ ██║╚██╔╝██║██╔══██╗██║     ██╔══██║██╔══██╗
██║  ██╗██║ ╚═╝ ██║██║  ██║███████╗██║  ██║██████╔╝
╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ 

Crafted with ❤️ by Kristofer Meio-Renn

Found this useful? Star the repo to show your support! Thank you!
GitHub: https://github.com/kmrlab
"""

import asyncio
import time
import signal
import sys
import logging
import os
from typing import Dict, Optional
from dataclasses import dataclass
import statistics

# Windows encoding setup
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from config import config

from src.glft_strategy import GLFTStrategy
from src.async_bybit_client import AsyncBybitClient
from src.execution_tracker import ExecutionTracker, ExecutionTrackerIntegration



@dataclass
class PerformanceMetrics:
    """Performance metrics for MFT strategy"""
    cycle_times: list
    jitter_values: list
    timing_violations: int
    parallel_operations: int
    
    def get_avg_cycle_time(self) -> float:
        return statistics.mean(self.cycle_times) if self.cycle_times else 0.0
    
    def get_jitter_std(self) -> float:
        return statistics.stdev(self.jitter_values) if len(self.jitter_values) > 1 else 0.0
    
    def get_timing_accuracy(self) -> float:
        """Percentage of cycles completed on time"""
        if not self.cycle_times:
            return 0.0
        on_time = sum(1 for t in self.cycle_times if t <= config.UPDATE_FREQUENCY * 1000)
        return (on_time / len(self.cycle_times)) * 100


class HighPrecisionTimer:
    """
    High-precision timer for MFT strategies
    Uses asyncio event loop for accurate timing
    """
    
    def __init__(self, interval: float):
        self.interval = interval
        self.next_execution = None
        self.drift_compensation = 0.0
        
    def reset(self):
        """Timer reset"""
        self.next_execution = time.perf_counter() + self.interval
        self.drift_compensation = 0.0
    
    async def wait_for_next_cycle(self) -> float:
        """
        Wait for next cycle with drift compensation
        Returns actual wait time
        """
        if self.next_execution is None:
            self.reset()
            return 0.0
        
        current_time = time.perf_counter()
        sleep_time = self.next_execution - current_time + self.drift_compensation
        
        # Minimum wait time
        if sleep_time > 0.001:
            await asyncio.sleep(sleep_time)
        
        # Calculate drift for next cycle
        actual_time = time.perf_counter()
        expected_time = self.next_execution
        drift = actual_time - expected_time
        
        # Drift compensation (no more than 10% of interval)
        max_compensation = self.interval * 0.1
        self.drift_compensation = max(-max_compensation, min(max_compensation, -drift * 0.5))
        
        # Plan next cycle
        self.next_execution += self.interval
        
        return sleep_time


class AdaptiveTimer:
    """Adaptive timer with dynamically changing interval"""
    
    def __init__(self, initial_interval: float):
        self.current_interval = initial_interval
        self.next_execution = None
        self.drift_compensation = 0.0
        
    def update_interval(self, new_interval: float):
        """Update interval with protection from sharp changes"""
        if abs(new_interval - self.current_interval) > config.FREQUENCY_UPDATE_THRESHOLD:
            self.current_interval = new_interval
            
    async def wait_for_next_cycle(self) -> float:
        """Wait with adaptive interval"""
        if self.next_execution is None:
            self.next_execution = time.perf_counter() + self.current_interval
            return 0.0
        
        current_time = time.perf_counter()
        sleep_time = self.next_execution - current_time + self.drift_compensation
        
        if sleep_time > 0.001:
            await asyncio.sleep(sleep_time)
        
        # Calculate drift for next cycle
        actual_time = time.perf_counter()
        expected_time = self.next_execution
        drift = actual_time - expected_time
        
        # Drift compensation
        max_compensation = self.current_interval * 0.1
        self.drift_compensation = max(-max_compensation, min(max_compensation, -drift * 0.5))
        
        # Plan next cycle with current interval
        self.next_execution += self.current_interval
        
        return sleep_time


class AsyncGLFTBot:
    """
    Asynchronous GLFT strategy trading bot
    Optimized for MFT with high timing precision
    """
    
    def __init__(self):
        self.strategy = GLFTStrategy()
        self.client = None  # Initialized in async context
        self.running = False
        self.logger = self._setup_logger()
        
        # Counters and statistics
        self.cycle_count = 0
        self.start_time = time.time()
        self.performance_metrics = PerformanceMetrics([], [], 0, 0)
        
        # Adaptive timer
        self.adaptive_timer = AdaptiveTimer(config.BASE_UPDATE_FREQUENCY)
        
        # Order execution tracking
        self.execution_tracker = ExecutionTracker()
        self.execution_integration = None  # Initialized later
        

        
        # Trading statistics
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'parallel_operations': 0
        }
        
        # Signal handler setup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self) -> logging.Logger:
        """Main logger setup"""
        logger = logging.getLogger('async_glft_bot')
        if not logger.handlers:
            # Console output
            console_handler = logging.StreamHandler()
            console_handler.setStream(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s | ASYNC_GLFT | %(levelname)s | %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
            
            # File output
            try:
                file_handler = logging.FileHandler(config.STRATEGY_LOG_FILE, encoding='utf-8')
                file_formatter = logging.Formatter(
                    '%(asctime)s | ASYNC_GLFT | %(levelname)s | %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(logging.DEBUG)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to create log file: {e}")
            
            logger.setLevel(logging.DEBUG)
        return logger
    
    def _signal_handler(self, signum, frame):
        """Signal handler for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def initialize(self) -> bool:
        """Asynchronous system initialization"""
        self.logger.info("=== STARTING ASYNC GLFT MARKET MAKING STRATEGY ===")
        self.logger.info(f"Symbol: {config.SYMBOL}")
        self.logger.info(f"Testnet: {config.TESTNET}")
        self.logger.info(f"Update frequency: {config.UPDATE_FREQUENCY} sec")
        self.logger.info(f"Production mode: {config.PRODUCTION_MODE}")
        self.logger.info(f"High precision timing: ENABLED")
        
        # Information about time decay
        if config.INVENTORY_TIME_DECAY_ENABLED:
            self.logger.info(f"Time decay: ENABLED ({config.INVENTORY_DECAY_TIME_MINUTES}min, {config.INVENTORY_DECAY_MULTIPLIER:.1f}x max)")
        else:
            self.logger.info(f"Time decay: DISABLED")
        
        # Information about Z-score volume adaptation
        if config.Z_SCORE_VOLUME_ADAPTATION_ENABLED:
            self.logger.info(f"Z-Score volume adaptation: ENABLED (intensity: {config.Z_SCORE_INTENSITY:.2f})")
        else:
            self.logger.info(f"Z-Score volume adaptation: DISABLED")
        
        # Client already initialized in context manager
        # Check time with server
        if not await self.client.validate_timestamp():
            self.logger.error("Server time synchronization error")
            return False
        
        # Get initial data in parallel
        initial_price, initial_position = await self.client.get_parallel_market_data()
        
        if initial_price is None:
            self.logger.error("Failed to get initial price")
            return False
        
        self.logger.info(f"Initial price: {initial_price}")
        self.logger.info(f"Initial position: {initial_position}")
        
        # Strategy initialization
        self.strategy.state.price = initial_price
        self.strategy.state.position = initial_position
        self.strategy.state.last_price = initial_price
        
        # Adaptive timer initialization
        self.adaptive_timer.next_execution = time.perf_counter() + self.adaptive_timer.current_interval
        
        # Execution tracking initialization
        self.execution_integration = ExecutionTrackerIntegration(self.execution_tracker)
        self.execution_tracker.start_new_period()
        

        
        self.logger.info("Async initialization completed successfully")
        return True
    
    async def place_orders_async(self, quotes: Dict[str, float], current_price: float) -> None:
        """Asynchronous order placement with execution tracking"""
        try:
            start_time = time.perf_counter()
            
            # Parallel placement of bid and ask orders
            bid_order_id, ask_order_id = await self.client.place_orders_parallel(
                bid_price=quotes['bid_price'],
                bid_size=quotes['bid_volume'],
                ask_price=quotes['ask_price'],
                ask_size=quotes['ask_volume']
            )
            
            # Track order placement for ExecutionTracker with real IDs
            if self.execution_integration:
                await self.execution_integration.track_order_placement(
                    self.client,
                    bid_price=quotes['bid_price'],
                    bid_size=quotes['bid_volume'],
                    ask_price=quotes['ask_price'],
                    ask_size=quotes['ask_volume'],
                    mid_price=current_price,
                    bid_order_id=bid_order_id,
                    ask_order_id=ask_order_id
                )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Statistics update
            self.stats['total_orders'] += 2
            self.stats['parallel_operations'] += 1
            
            if bid_order_id:
                self.stats['successful_orders'] += 1
            else:
                self.stats['failed_orders'] += 1
                
            if ask_order_id:
                self.stats['successful_orders'] += 1
            else:
                self.stats['failed_orders'] += 1
            
            # Result logging
            if bid_order_id and ask_order_id:
                self.logger.info(
                    f"ORDERS PLACED: "
                    f"BID {quotes['bid_volume']:.1f}@{quotes['bid_price']:.2f} | "
                    f"ASK {quotes['ask_volume']:.1f}@{quotes['ask_price']:.2f} | "
                    f"Time: {execution_time:.1f}ms"
                )
            elif bid_order_id or ask_order_id:
                self.logger.warning(f"Only one order placed out of two (Time: {execution_time:.1f}ms)")
            else:
                self.logger.warning(f"Failed to place any orders (Time: {execution_time:.1f}ms)")
                
        except Exception as e:
            self.logger.error(f"Async order placement error: {e}")
            self.stats['failed_orders'] += 2
    
    def log_performance_stats(self) -> None:
        """MFT performance statistics logging"""
        uptime = time.time() - self.start_time
        success_rate = (self.stats['successful_orders'] / max(1, self.stats['total_orders'])) * 100
        
        strategy_info = self.strategy.get_strategy_info()
        
        # Performance metrics
        avg_cycle_time = self.performance_metrics.get_avg_cycle_time()
        jitter_std = self.performance_metrics.get_jitter_std()
        timing_accuracy = self.performance_metrics.get_timing_accuracy()
        
        # Z-score statistics
        z_score_stats = ""
        if config.Z_SCORE_VOLUME_ADAPTATION_ENABLED:
            z_score = self.strategy.state.current_z_score
            z_category = strategy_info.get('z_score_category', 'N/A')
            z_active = strategy_info.get('z_volume_adaptations_active', False)
            z_score_stats = f" | Z-Score: {z_score:+.2f}({z_category}) {'ACTIVE' if z_active else 'INACTIVE'}"
        
        self.logger.info(
            f"ASYNC STATS: Uptime: {uptime/60:.1f}min | "
            f"Cycles: {self.cycle_count} | "
            f"Orders: {self.stats['total_orders']} | "
            f"Success: {success_rate:.1f}% | "
            f"Parallel: {self.stats['parallel_operations']} | "
            f"Avg Cycle: {avg_cycle_time:.1f}ms | "
            f"Jitter: {jitter_std:.1f}ms | "
            f"Timing Violations: {self.performance_metrics.timing_violations}{z_score_stats}"
        )
    
    async def run_cycle(self) -> bool:
        """Execute one asynchronous strategy cycle"""
        try:
            cycle_start_time = time.perf_counter()
            
            # 1. Parallel market data retrieval
            current_price, current_position = await self.client.get_parallel_market_data()
            
            if current_price is None:
                self.logger.error("Failed to get current price")
                return False
            
            # 2. Strategy update and quote calculation
            quotes = self.strategy.update_state(current_price, current_position)
            
            # 3. Adaptive frequency calculation
            performance_data = {
                'avg_cycle_time_ms': self.performance_metrics.get_avg_cycle_time()
            }
            new_frequency = self.strategy.calculate_adaptive_frequency(performance_data)
            
            # 4. Update timer if frequency changed significantly
            self.adaptive_timer.update_interval(new_frequency)
            
            # 5. Logging depending on mode
            if config.PRODUCTION_MODE:
                self._log_production_cycle(cycle_start_time, current_price, current_position, quotes)
            else:
                self._log_detailed_cycle(cycle_start_time, current_price, current_position, quotes)
            
            # 6. Asynchronous order placement
            await self.place_orders_async(quotes, current_price)
            
            # 6.1. Execution tracking (every 5 cycles for optimization)
            if self.cycle_count % 5 == 0 and self.execution_integration:
                try:
                    await self.execution_integration.fetch_and_track_executions(self.client)
                except Exception as e:
                    self.logger.warning(f"Error tracking executions: {e}")
            
            # 6.2. NEW: Pass current strategy parameters to ExecutionTracker
            if self.execution_tracker:
                current_strategy_params = {
                    'A_BASE': config.A_BASE,
                    'K_BASE': config.K_BASE,
                    'GAMMA': config.GAMMA,
                    'BASE_VOLATILITY': config.BASE_VOLATILITY,
                    'INVENTORY_FACTOR': config.INVENTORY_FACTOR
                }
                self.execution_tracker.set_current_strategy_params(
                    current_strategy_params, 
                    self.strategy.state.cycle_count
                )


            # 7. Performance metrics update
            cycle_end_time = time.perf_counter()
            cycle_duration_ms = (cycle_end_time - cycle_start_time) * 1000
            
            self.performance_metrics.cycle_times.append(cycle_duration_ms)
            
            # Limit metrics history size
            if len(self.performance_metrics.cycle_times) > 1000:
                self.performance_metrics.cycle_times = self.performance_metrics.cycle_times[-500:]
            
            # Check timing violations
            if cycle_duration_ms > config.UPDATE_FREQUENCY * 1000 * 0.9:
                self.performance_metrics.timing_violations += 1
                self.logger.warning(f"Timing violation: cycle took {cycle_duration_ms:.1f}ms")
            
            self.cycle_count += 1
            
            # 8. Periodic statistics
            if self.cycle_count % (5 * 60 // config.UPDATE_FREQUENCY) == 0:
                self.log_performance_stats()
                
                # Log execution metrics every 5 minutes
                if self.execution_tracker:
                    self.execution_tracker.log_period_summary()
                

            
            return True
            
        except Exception as e:
            self.logger.error(f"Async strategy cycle error: {e}")
            return False
    
    def _log_production_cycle(self, cycle_start_time: float, current_price: float, 
                            current_position: float, quotes: Dict) -> None:
        """Minimal logging for production"""
        current_time_str = time.strftime("%H:%M:%S", time.localtime(cycle_start_time))
        spread_pct = (quotes['ask_price'] - quotes['bid_price']) / current_price * 100
        
        # Information about time decay (если включено)
        decay_info = ""
        if config.INVENTORY_TIME_DECAY_ENABLED:
            strategy_info = self.strategy.get_strategy_info()
            if 'time_decay_multiplier' in strategy_info:
                time_mult = strategy_info['time_decay_multiplier']
                if time_mult > 1.01:  # Показываем только если есть значимое усиление
                    decay_info = f" | Decay: {time_mult:.2f}x"
        
        # Информация об адаптивной частоте для продакшн лога
        frequency_info = ""
        current_freq = self.strategy.state.smoothed_frequency
        if abs(current_freq - config.BASE_UPDATE_FREQUENCY) > 0.5:  # Показываем только при значимых изменениях
            frequency_info = f" | Freq: {current_freq:.1f}s"
        
        # Расчет асимметрии спредов для продакшн лога
        bid_spread = abs(current_price - quotes['bid_price'])
        ask_spread = abs(quotes['ask_price'] - current_price)
        asymmetry_info = ""
        if abs(bid_spread - ask_spread) > 0.01:  # Показываем только если есть значимая асимметрия
            asymmetry_info = f" | B:{bid_spread:.3f}/A:{ask_spread:.3f}"
        
        # Z-score информация для продакшн лога
        z_score_info = ""
        if config.Z_SCORE_VOLUME_ADAPTATION_ENABLED:
            z_score = quotes.get('z_score', 0.0)
            z_active = quotes.get('z_adjustment_applied', False)
            z_conflict = quotes.get('z_conflict_detected', False)
            
            if abs(z_score) > 0.5:  # Показываем только значимые Z-score
                if z_conflict:
                    z_status = "CONF"
                elif z_active:
                    z_status = "ACT"
                else:
                    z_status = "INACT"
                z_score_info = f" | Z:{z_score:+.2f}({z_status})"
        
        self.logger.info(
            f"ASYNC CYCLE {current_time_str} | "
            f"Price: {current_price:.2f} | Pos: {current_position:.1f} | "
            f"BID: {quotes['bid_price']:.2f} | ASK: {quotes['ask_price']:.2f} | "
            f"Spread: {spread_pct:.3f}% | Vol: {self.strategy.state.volatility:.4f}{decay_info}{frequency_info}{asymmetry_info}{z_score_info}"
        )
    
    def _log_detailed_cycle(self, cycle_start_time: float, current_price: float,
                          current_position: float, quotes: Dict) -> None:
        """Detailed logging for development"""
        strategy_info = self.strategy.get_detailed_metrics()
        
        self.logger.info("=" * 85)
        self.logger.info("ASYNC CYCLE")
        self.logger.info("-" * 85)
        
        current_time_str = time.strftime("%H:%M:%S", time.localtime(cycle_start_time))
        self.logger.info(f"Time: {current_time_str}    Cycle: {self.cycle_count}")
        self.logger.info(f"Market Price:     {current_price:>10.2f}    Position:         {current_position:>8.1f}")
        self.logger.info(f"Volatility:       {self.strategy.state.volatility:>10.4f}    Shock Mult:       {self.strategy.state.shock_multiplier:>8.3f}")
        
        # Information about time decay
        if config.INVENTORY_TIME_DECAY_ENABLED:
            full_strategy_info = self.strategy.get_strategy_info()
            if 'time_decay_multiplier' in full_strategy_info:
                time_mult = full_strategy_info['time_decay_multiplier']
                time_in_pos = full_strategy_info.get('time_in_position_seconds', 0)
                decay_progress = min(1.0, time_in_pos / (config.INVENTORY_DECAY_TIME_MINUTES * 60)) if time_in_pos > 0 else 0.0
                max_mult = 1.0 + config.INVENTORY_DECAY_MULTIPLIER
                self.logger.info(f"Time Decay Mult:  {time_mult:>10.3f}    Time in Pos:      {time_in_pos/60:>8.1f}min")
                self.logger.info(f"Decay Progress:   {decay_progress:>10.1%}    Max Multiplier:   {max_mult:>8.1f}x")
        
        # Асинхронные метрики
        if self.performance_metrics.cycle_times:
            avg_cycle = statistics.mean(self.performance_metrics.cycle_times[-10:])  # Последние 10 циклов
            self.logger.info(f"Avg Cycle Time:   {avg_cycle:>10.1f}ms   Parallel Ops:     {self.stats['parallel_operations']:>8d}")
        
        self.logger.info("-" * 45)
        self.logger.info("GLFT PARAMETERS:")
        self.logger.info(f"A (adaptive):     {self.strategy.state.a_adaptive:>10.3f}    K (adaptive):     {self.strategy.state.k_adaptive:>8.2f}")
        self.logger.info(f"c1 component:     {strategy_info['c1']:>10.6f}    c2 component:     {strategy_info['c2']:>8.6f}")
        
        self.logger.info("-" * 45)
        self.logger.info("SPREAD ANALYSIS:")
        # Расчет полного текущего спреда
        current_full_spread = quotes['ask_price'] - quotes['bid_price']
        spread_pct = (current_full_spread / current_price) * 100
        
        # Расчет асимметричных спредов от mid price
        bid_spread_from_mid = abs(current_price - quotes['bid_price'])
        ask_spread_from_mid = abs(quotes['ask_price'] - current_price)
        
        self.logger.info(f"Base Spread:      {strategy_info['base_spread']:>10.4f}    Final Spread:     {strategy_info['final_spread']:>8.4f}")
        self.logger.info(f"Current Spread:   {current_full_spread:>10.4f}    Spread %:         {spread_pct:>8.3f}%")
        self.logger.info(f"Expansion Ratio:  {strategy_info['spread_expansion']:>10.4f}    Expansion Amount: {strategy_info['spread_expansion_amount']:>8.4f}")
        self.logger.info(f"Half Spread:      {quotes['half_spread']:>10.4f}    Inventory Skew:   {quotes['skew']:>8.4f}")
        
        # НОВОЕ: Детальный анализ влияния волатильности
        self.logger.info("-" * 45)
        self.logger.info("VOLATILITY IMPACT ANALYSIS:")
        self.logger.info(f"Vol Adjustment:   {strategy_info['volatility_adjustment']:>10.4f}    Vol Ratio:        {strategy_info['vol_ratio']:>8.3f}")
        self.logger.info(f"C1 Component:     {strategy_info['c1_component']:>10.4f}    Vol Component:    {strategy_info['volatility_component']:>8.4f}")
        self.logger.info(f"Total from C1+Vol:{strategy_info['c1_component'] + strategy_info['volatility_component']:>10.4f}    Base Spread:      {strategy_info['base_spread']:>8.4f}")
        
        # НОВОЕ: Информация об адаптивной частоте
        self.logger.info("-" * 45)
        self.logger.info("ADAPTIVE FREQUENCY ANALYSIS:")
        current_frequency = self.strategy.state.smoothed_frequency
        base_frequency = config.BASE_UPDATE_FREQUENCY
        frequency_change = ((current_frequency - base_frequency) / base_frequency) * 100
        
        # Расчет факторов для отображения
        vol_ratio = self.strategy.state.volatility / config.BASE_VOLATILITY
        f_volatility = vol_ratio ** config.FREQUENCY_VOLATILITY_SENSITIVITY
        position_ratio = abs(self.strategy.state.position) / config.MAX_POSITION
        f_position = 1.0 + position_ratio * config.FREQUENCY_POSITION_SENSITIVITY
        f_shock = self.strategy.state.shock_multiplier ** config.FREQUENCY_SHOCK_SENSITIVITY
        activity_multiplier = f_volatility * f_position * f_shock
        
        self.logger.info(f"Current Frequency:{current_frequency:>10.2f}s   Base Frequency:   {base_frequency:>8.1f}s")
        self.logger.info(f"Frequency Change: {frequency_change:>10.1f}%    Activity Level:   {activity_multiplier:>8.2f}x")
        self.logger.info(f"Vol Factor:       {f_volatility:>10.2f}     Pos Factor:       {f_position:>8.2f}")
        self.logger.info(f"Shock Factor:     {f_shock:>10.2f}     Timer Interval:   {self.adaptive_timer.current_interval:>8.2f}s")
        
        # НОВОЕ: Асимметричные спреды от mid price
        self.logger.info("-" * 45)
        self.logger.info("ASYMMETRIC SPREADS FROM MID:")
        self.logger.info(f"BID Spread:       {bid_spread_from_mid:>10.4f}    ASK Spread:       {ask_spread_from_mid:>8.4f}")
        
        # Показываем конкретные значения сжатия/расширения спредов
        if abs(quotes['skew']) > 0.001:
            # Расчет теоретических симметричных спредов (без skew)
            theoretical_spread = quotes['half_spread']  # Это final_half_spread без skew
            
            # Фактические отклонения от симметричного спреда
            bid_deviation = bid_spread_from_mid - theoretical_spread
            ask_deviation = ask_spread_from_mid - theoretical_spread
            
            self.logger.info(f"Position Effect:  BID {bid_deviation:+.4f}    ASK {ask_deviation:+.4f}")
        else:
            self.logger.info(f"Position Effect:  {'SYMMETRIC':>10s}    {'(no position)':>17s}")
        
        # Детали временного decay для inventory skew
        if config.INVENTORY_TIME_DECAY_ENABLED:
            full_strategy_info = self.strategy.get_strategy_info()
            if 'time_decay_multiplier' in full_strategy_info:
                time_mult = full_strategy_info['time_decay_multiplier']
                base_skew = quotes['skew'] / time_mult if time_mult > 0 else quotes['skew']
                max_mult = 1.0 + config.INVENTORY_DECAY_MULTIPLIER
                time_minutes = config.INVENTORY_DECAY_TIME_MINUTES
                self.logger.info(f"Base Inv Skew:    {base_skew:>10.4f}    Time Multiplier:  {time_mult:>8.3f}")
                self.logger.info(f"Config Max Mult:  {max_mult:>10.1f}x   Config Time:      {time_minutes:>8d}min")
        
        # Z-SCORE АДАПТАЦИЯ ОБЪЕМОВ
        if config.Z_SCORE_VOLUME_ADAPTATION_ENABLED:
            self.logger.info("-" * 45)
            self.logger.info("Z-SCORE VOLUME ADAPTATION:")
            
            z_score = quotes.get('z_score', 0.0)
            z_category = strategy_info.get('z_score_category', 'N/A')
            z_active = quotes.get('z_adjustment_applied', False)
            z_conflict = quotes.get('z_conflict_detected', False)
            
            self.logger.info(f"Z-Score:          {z_score:>10.3f}    Category:         {z_category:>8s}")
            self.logger.info(f"Intensity:        {config.Z_SCORE_INTENSITY:>10.2f}    Adaptations:      {'ACTIVE' if z_active else 'INACTIVE':>8s}")
            
            if z_active and 'volume_adaptations' in quotes:
                vol_adapt = quotes['volume_adaptations']
                adj_info = vol_adapt.get('adjustment_info', {})
                
                bid_mult = adj_info.get('bid_multiplier', 1.0)
                ask_mult = adj_info.get('ask_multiplier', 1.0)
                mode = adj_info.get('mode', 'N/A')
                strength = adj_info.get('correction_strength', 0.0)
                
                self.logger.info(f"BID Multiplier:   {bid_mult:>10.3f}    ASK Multiplier:   {ask_mult:>8.3f}")
                self.logger.info(f"Correction Mode:  {mode:>10s}    Strength:         {strength:>8.2f}")
                
                if z_conflict:
                    inv_signal = adj_info.get('inventory_signal', 0)
                    z_signal = adj_info.get('z_score_signal', 0)
                    self.logger.info(f"CONFLICT:         Inventory={inv_signal:>+2.0f}     Z-score={z_signal:>+2.0f}      {'REDUCED STRENGTH':>8s}")
            
            # Расчет влияния на объемы
            base_bid_vol = config.BASE_ORDER_SIZE
            base_ask_vol = config.BASE_ORDER_SIZE
            final_bid_vol = quotes['bid_volume']
            final_ask_vol = quotes['ask_volume']
            
            bid_change = ((final_bid_vol - base_bid_vol) / base_bid_vol) * 100
            ask_change = ((final_ask_vol - base_ask_vol) / base_ask_vol) * 100
            
            self.logger.info(f"Volume Changes:   BID {bid_change:>+7.1f}%    ASK {ask_change:>+7.1f}%    Base: {base_bid_vol:>6.1f}")
        
        self.logger.info("-" * 45)
        self.logger.info("ASYNC QUOTES & ORDERS:")
        self.logger.info(f"BID: {quotes['bid_volume']:>6.1f} @ {quotes['bid_price']:>8.2f}    Half Spread:      {quotes['half_spread']:>8.4f}")
        self.logger.info(f"ASK: {quotes['ask_volume']:>6.1f} @ {quotes['ask_price']:>8.2f}    Inventory Skew:   {quotes['skew']:>8.4f}")
    
    async def run(self) -> None:
        """Основной асинхронный цикл торгового бота"""
        # Инициализация в асинхронном контексте
        async with AsyncBybitClient(use_testnet=config.TESTNET) as client:
            self.client = client
            
            if not await self.initialize():
                self.logger.error("Async initialization error, shutting down")
                return
            
            self.running = True
            self.logger.info("Starting async trading cycle with high precision timing...")
            
            try:
                cycle_count = 0
                
                while self.running:
                    # Адаптивное ожидание следующего цикла
                    await self.adaptive_timer.wait_for_next_cycle()
                    
                    # Выполнение цикла стратегии
                    cycle_success = await self.run_cycle()
                    
                    if not cycle_success:
                        self.logger.error("Critical async cycle error, shutting down...")
                        break
                    
                    cycle_count += 1
                    
                    # Расчет jitter для метрик (адаптированный для переменного интервала)
                    current_time = time.perf_counter()
                    expected_time = self.start_time + cycle_count * self.adaptive_timer.current_interval
                    jitter = abs(current_time - expected_time) * 1000
                    
                    self.performance_metrics.jitter_values.append(jitter)
                    if len(self.performance_metrics.jitter_values) > 100:
                        self.performance_metrics.jitter_values = self.performance_metrics.jitter_values[-50:]
            
            except KeyboardInterrupt:
                self.logger.info("Received user interrupt signal")
            except Exception as e:
                self.logger.error(f"Unexpected async error: {e}")
            finally:
                await self.shutdown()
    
    async def shutdown(self) -> None:
        """Корректное асинхронное завершение работы"""
        self.logger.info("Shutting down async bot...")
        
        # Отмена всех активных ордеров
        if self.client:
            try:
                await self.client.cancel_all_orders()
                self.logger.info("All orders cancelled")
            except Exception as e:
                self.logger.error(f"Order cancellation error: {e}")
        

        # Финальная статистика
        self.log_performance_stats()
        
        # Статистика API производительности
        if self.client:
            api_stats = self.client.get_performance_stats()
            if api_stats:
                self.logger.info(f"API Performance: Avg: {api_stats['avg_api_time_ms']}ms, "
                               f"Max: {api_stats['max_api_time_ms']}ms, "
                               f"Calls: {api_stats['total_api_calls']}")
        
        self.logger.info("=== ASYNC GLFT STRATEGY SHUTDOWN COMPLETE ===")


async def main():
    """Asynchronous program entry point"""
    print("ASYNC GLFT Market-making strategy for Bybit")
    print("Optimized for MFT with high-precision timers")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    bot = AsyncGLFTBot()
    await bot.run()


if __name__ == "__main__":
    # Запуск асинхронного приложения
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Critical error: {e}")