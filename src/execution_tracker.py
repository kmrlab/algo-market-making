import time
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class TradeRecord:
    """Trade execution record"""
    timestamp: float
    exec_id: str
    side: str  # 'Buy' or 'Sell'
    size: float
    price: float
    order_timestamp: float  # Order placement timestamp
    quoted_spread: float    # Spread at order placement time
    mid_price: float       # Mid price at order placement time
    
    @property
    def fill_time_ms(self) -> float:
        """Fill time in milliseconds"""
        return (self.timestamp - self.order_timestamp) * 1000
    
    @property
    def realized_spread(self) -> float:
        """Realized spread (difference from mid price)"""
        if self.side == 'Buy':
            return self.mid_price - self.price  # Positive if bought cheaper than mid
        else:
            return self.price - self.mid_price  # Positive if sold higher than mid

@dataclass
class ExecutionMetrics:
    """Execution metrics for a period"""
    trades_count: int
    avg_fill_time_ms: float
    spread_efficiency: float
    fill_rate: float
    avg_slippage: float
    total_volume: float


class ExecutionTracker:
    """
    Tracking order execution quality metrics
    Integrates with AsyncBybitClient to get trade data
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Current period data
        self.period_trades: List[TradeRecord] = []
        self.period_orders: Dict[str, Dict] = {}  # order_id -> order_data
        self.period_start_time = 0
        
        # Rolling windows for analysis
        self.recent_fill_times = deque(maxlen=100)
        self.recent_spreads = deque(maxlen=100)
        
        # Last metrics cache
        self.last_metrics: Optional[ExecutionMetrics] = None
        
        # Integration with trading strategy
        self.current_strategy_params: Dict[str, float] = {}
        self.current_cycle_number: int = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Logger setup with file saving"""
        logger = logging.getLogger('execution_tracker')
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s | EXECUTION | %(levelname)s | %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            from pathlib import Path
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(logs_dir / "execution_tracker.log", encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.setLevel(logging.INFO)
        return logger
        
    def start_new_period(self):
        """Start new tracking period"""
        self.logger.info("Starting new execution tracking period")
        
        # Save metrics from previous period
        if self.period_trades:
            self.last_metrics = self._calculate_metrics()
            self.logger.info(f"Previous period metrics: {self.last_metrics}")
        
        # Clear period data
        self.period_trades = []
        self.period_orders = {}
        self.period_start_time = time.time()
        
    def record_order_placement(self, order_id: str, order_data: Dict):
        """Record order placement"""
        self.period_orders[order_id] = {
            'timestamp': time.time(),
            'side': order_data.get('side', 'unknown'),
            'size': order_data.get('qty', 0),
            'price': order_data.get('price', 0),
            'quoted_spread': order_data.get('quoted_spread', 0),
            'mid_price': order_data.get('mid_price', 0)
        }
        
        self.logger.debug(f"Recorded order placement: {order_id}")
        

    def set_execution_filter(self, last_processed_exec_id: str = None, 
                           last_processed_timestamp: float = 0.0):
        """Sets filter to process only new trades"""
        self.last_processed_exec_id = last_processed_exec_id
        self.last_processed_timestamp = last_processed_timestamp
        
        if last_processed_exec_id:
            self.logger.info(f"Execution filter set: last processed {last_processed_exec_id} at {last_processed_timestamp}")
        else:
            self.logger.info("Execution filter set: process all new executions")
    
    def should_process_execution(self, execution) -> bool:
        """Checks if this trade should be processed"""
        exec_id = execution.get('execId', '')
        exec_time = float(execution.get('execTime', 0)) / 1000
        
        # If filter by last processed trade is set
        if hasattr(self, 'last_processed_timestamp') and self.last_processed_timestamp > 0:
            # Process only trades newer than last processed
            if exec_time <= self.last_processed_timestamp:
                return False
        
        # If period start time exists, don't process older trades
        if hasattr(self, 'period_start_time') and self.period_start_time > 0:
            if exec_time < self.period_start_time:
                return False
        
        return True
    
    def update_last_processed_execution(self, execution):
        """Updates information about last processed trade"""
        exec_id = execution.get('execId', '')
        exec_time = float(execution.get('execTime', 0)) / 1000
        
        if not hasattr(self, 'last_processed_exec_id'):
            self.last_processed_exec_id = None
        if not hasattr(self, 'last_processed_timestamp'):
            self.last_processed_timestamp = 0.0
        if not hasattr(self, 'total_processed_trades'):
            self.total_processed_trades = 0
        
        # Update only if this is a newer trade
        if exec_time > self.last_processed_timestamp:
            self.last_processed_exec_id = exec_id
            self.last_processed_timestamp = exec_time
            self.total_processed_trades += 1
            
            self.logger.debug(f"Updated last processed: {exec_id} at {exec_time}")

    def set_current_strategy_params(self, params: Dict[str, float], cycle_num: int):
        """Set current strategy parameters for linking to trades"""
        self.current_strategy_params = params.copy()
        self.current_cycle_number = cycle_num
        self.logger.debug(f"Updated strategy params for cycle {cycle_num}")



    def record_trade_execution(self, trade_data: Dict):
        """Record trade execution with filtering"""
        exec_id = trade_data.get('execId', '')
        order_id = trade_data.get('orderId', '')
        
        # Check if this trade should be processed
        if not self.should_process_execution(trade_data):
            self.logger.debug(f"Trade {exec_id} filtered out (already processed or too old)")
            return
        
        # Find corresponding order
        order_info = self.period_orders.get(order_id)
        if not order_info:
            self.logger.warning(f"Trade {exec_id} without matching order {order_id}")
            return
            
        # Create trade record
        trade_record = TradeRecord(
            timestamp=float(trade_data.get('execTime', 0)) / 1000,  # Bybit time in ms
            exec_id=exec_id,
            side=trade_data.get('side', 'unknown'),
            size=float(trade_data.get('execQty', 0)),
            price=float(trade_data.get('execPrice', 0)),
            order_timestamp=order_info['timestamp'],
            quoted_spread=order_info['quoted_spread'],
            mid_price=order_info['mid_price']
        )
        
        self.period_trades.append(trade_record)
        
        # Update rolling windows
        self.recent_fill_times.append(trade_record.fill_time_ms)
        self.recent_spreads.append(trade_record.realized_spread)
        
        # Update last processed trade information
        self.update_last_processed_execution(trade_data)
        
        self.logger.debug(f"Recorded trade execution: {exec_id}, fill_time: {trade_record.fill_time_ms:.1f}ms, PnL: {trade_record.realized_spread * trade_record.size:.4f}")
        
    def get_period_metrics(self) -> ExecutionMetrics:
        """Get metrics for current period"""
        return self._calculate_metrics()
        
    def _calculate_metrics(self) -> ExecutionMetrics:
        """Calculate execution metrics"""
        if not self.period_trades:
            return ExecutionMetrics(
                trades_count=0,
                avg_fill_time_ms=0.0,
                spread_efficiency=0.0,
                fill_rate=0.0,
                avg_slippage=0.0,
                total_volume=0.0
            )
            
        trades_count = len(self.period_trades)
        total_orders = len(self.period_orders)
        
        # Fill time
        fill_times = [trade.fill_time_ms for trade in self.period_trades]
        avg_fill_time_ms = np.mean(fill_times)
        
        # Spread efficiency (fraction of spread captured by strategy)
        spread_ratios = []
        for trade in self.period_trades:
            if trade.quoted_spread > 0:
                # Efficiency = realized_spread / quoted_spread
                efficiency = trade.realized_spread / trade.quoted_spread
                # Limit from 0 to 1 (100% efficiency)
                spread_ratios.append(max(0.0, min(1.0, efficiency)))
        
        spread_efficiency = np.mean(spread_ratios) if spread_ratios else 0.0
        
        # Fill rate (percentage of filled orders)
        fill_rate = trades_count / max(1, total_orders)
        
        # Slippage (deviation from quoted price)
        slippages = []
        for trade in self.period_trades:
            order_info = self.period_orders.get(trade.exec_id)
            if order_info:
                expected_price = order_info['price']
                slippage = abs(trade.price - expected_price) / expected_price
                slippages.append(slippage)
        
        avg_slippage = np.mean(slippages) if slippages else 0.0
        
        # Total volume
        total_volume = sum(trade.size for trade in self.period_trades)
        
        return ExecutionMetrics(
            trades_count=trades_count,
            avg_fill_time_ms=avg_fill_time_ms,
            spread_efficiency=spread_efficiency,
            fill_rate=fill_rate,
            avg_slippage=avg_slippage,
            total_volume=total_volume
        )
        
    def get_real_time_metrics(self) -> Dict:
        """Get real-time metrics (rolling windows)"""
        if not self.recent_fill_times or not self.recent_spreads:
            return {
                'recent_avg_fill_time_ms': 0.0,
                'recent_spread_efficiency': 0.0,
                'fill_time_trend': 'stable'
            }
            
        recent_fill_time = np.mean(list(self.recent_fill_times))
        recent_spread_eff = np.mean([max(0.0, s) for s in self.recent_spreads])
        
        # Fill time trend (compare last 20 vs previous 20)
        if len(self.recent_fill_times) >= 40:
            recent_20 = list(self.recent_fill_times)[-20:]
            prev_20 = list(self.recent_fill_times)[-40:-20]
            
            recent_avg = np.mean(recent_20)
            prev_avg = np.mean(prev_20)
            
            if recent_avg > prev_avg * 1.1:
                trend = 'degrading'
            elif recent_avg < prev_avg * 0.9:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
            
        return {
            'recent_avg_fill_time_ms': recent_fill_time,
            'recent_spread_efficiency': recent_spread_eff,
            'fill_time_trend': trend,
            'data_points': len(self.recent_fill_times)
        }
        
    def get_execution_quality_score(self) -> float:
        """
        Overall execution quality score (0-1)
        Combines all metrics into single assessment
        """
        metrics = self.get_period_metrics()
        
        if metrics.trades_count == 0:
            return 0.0
            
        # Normalize metrics (higher is better)
        
        # Fill rate (0-1, optimal close to 1)
        fill_score = metrics.fill_rate
        
        # Spread efficiency (0-1, optimal close to 1)
        spread_score = metrics.spread_efficiency
        
        # Fill time (normalize: <100ms=1.0, >1000ms=0.0)
        fill_time_score = max(0.0, min(1.0, (1000 - metrics.avg_fill_time_ms) / 900))
        
        # Slippage (normalize: <0.1%=1.0, >1%=0.0)
        slippage_score = max(0.0, min(1.0, (0.01 - metrics.avg_slippage) / 0.009))
        
        # Weighted combination
        quality_score = (
            fill_score * 0.3 +          # 30% - important to fill orders
            spread_score * 0.4 +        # 40% - important to capture spread
            fill_time_score * 0.2 +     # 20% - execution speed
            slippage_score * 0.1        # 10% - minimal slippage
        )
        
        return quality_score
        
    def log_period_summary(self):
        """Log period summary"""
        metrics = self.get_period_metrics()
        quality_score = self.get_execution_quality_score()
        
        self.logger.info("=" * 60)
        self.logger.info("EXECUTION METRICS SUMMARY")
        self.logger.info("-" * 60)
        self.logger.info(f"Trades Count:        {metrics.trades_count}")
        self.logger.info(f"Total Volume:        {metrics.total_volume:.1f}")
        self.logger.info(f"Fill Rate:           {metrics.fill_rate:.1%}")
        self.logger.info(f"Avg Fill Time:       {metrics.avg_fill_time_ms:.1f}ms")
        self.logger.info(f"Spread Efficiency:   {metrics.spread_efficiency:.1%}")
        self.logger.info(f"Avg Slippage:        {metrics.avg_slippage:.3%}")
        self.logger.info(f"Quality Score:       {quality_score:.1%}")
        self.logger.info("=" * 60)


# Integration with AsyncBybitClient
class ExecutionTrackerIntegration:
    """ExecutionTracker integration with trading cycle"""
    
    def __init__(self, execution_tracker: ExecutionTracker):
        self.tracker = execution_tracker
        
    async def track_order_placement(self, client, bid_price: float, bid_size: float, 
                                  ask_price: float, ask_size: float, mid_price: float, 
                                  bid_order_id: str = None, ask_order_id: str = None):
        """Track order placement with real order_ids"""
        
        quoted_spread = ask_price - bid_price
        
        # Record data about placed orders with real IDs
        if bid_order_id:
            bid_order_data = {
                'side': 'Buy',
                'qty': bid_size,
                'price': bid_price,
                'quoted_spread': quoted_spread,
                'mid_price': mid_price
            }
            self.tracker.record_order_placement(bid_order_id, bid_order_data)
            self.tracker.logger.info(f"Tracked BID order: {bid_order_id} @ {bid_price}")
        
        if ask_order_id:
            ask_order_data = {
                'side': 'Sell', 
                'qty': ask_size,
                'price': ask_price,
                'quoted_spread': quoted_spread,
                'mid_price': mid_price
            }
            self.tracker.record_order_placement(ask_order_id, ask_order_data)
            self.tracker.logger.info(f"Tracked ASK order: {ask_order_id} @ {ask_price}")
        
    async def fetch_and_track_executions(self, client):
        """Fetch and track executions from exchange"""
        
        try:
            # Get latest executions
            executions_response = await client._make_request("GET", "/v5/execution/list", {
                "category": "linear",
                "symbol": "SOLUSDT",
                "limit": "50"
            })
            
            # Check that response is not None
            if executions_response is None:
                self.tracker.logger.warning("No response received when fetching executions")
                return
                
            if executions_response.get('retCode') == 0:
                executions = executions_response.get('result', {}).get('list', [])
                
                for execution in executions:
                    self.tracker.record_trade_execution(execution)
            else:
                self.tracker.logger.warning(f"API returned error when fetching executions: {executions_response.get('retMsg', 'Unknown error')}")
                    
        except Exception as e:
            self.tracker.logger.error(f"Error fetching executions: {e}")