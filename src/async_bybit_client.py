import asyncio
import aiohttp
import time
import logging
import hmac
import hashlib
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


@dataclass
class MarketData:
    """Market data structure"""
    price: float
    bid_price: float
    ask_price: float
    timestamp: float


@dataclass
class PositionData:
    """Position data structure"""
    size: float
    side: str
    unrealized_pnl: float
    timestamp: float


class AsyncBybitClient:
    """
    High-performance asynchronous Bybit API client
    Optimized for MFT strategies with precise timing
    """
    
    def __init__(self, use_testnet: bool = False):
        self.logger = self._setup_logger()
        self.use_testnet = use_testnet
        
        # URL endpoints
        if use_testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
            
        # Data caching for optimization
        self.price_cache = None
        self.price_cache_time = 0
        self.position_cache = None
        self.position_cache_time = 0
        
        # Session for connection reuse
        self.session = None
        
        # Performance statistics
        self.api_call_times = []
        self.parallel_calls_count = 0
        
        self.logger.info(f"Async Bybit client initialized (testnet={use_testnet})")
    
    def _setup_logger(self) -> logging.Logger:
        """Logger setup"""
        logger = logging.getLogger('async_bybit')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | ASYNC_BYBIT | %(levelname)s | %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def __aenter__(self):
        """Asynchronous context manager - enter"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(
                limit=100,  # Максимум соединений
                limit_per_host=50,  # На один хост
                keepalive_timeout=60,  # Время жизни соединения
                enable_cleanup_closed=True
            )
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Asynchronous context manager - exit"""
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, params: str, timestamp: str) -> str:
        """Generate signature for API request"""
        param_str = timestamp + config.API_KEY + "5000" + params
        return hmac.new(
            config.API_SECRET.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _prepare_headers(self, params: str) -> Dict[str, str]:
        """Prepare headers for API request"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(params, timestamp)
        
        return {
            "X-BAPI-API-KEY": config.API_KEY,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Universal method for API requests"""
        if not self.session:
            self.logger.error("Session not initialized. Use async context manager.")
            return None
            
        start_time = time.time()
        data = None  # Инициализируем переменную data
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method == "GET":
                if params:
                    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                    url += f"?{query_string}"
                    headers = self._prepare_headers(query_string)
                else:
                    headers = self._prepare_headers("")
                
                async with self.session.get(url, headers=headers) as response:
                    data = await response.json()
                    
            elif method == "POST":
                params_str = json.dumps(params) if params else "{}"
                headers = self._prepare_headers(params_str)
                
                async with self.session.post(url, headers=headers, data=params_str) as response:
                    data = await response.json()
            else:
                self.logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            # Record execution time
            execution_time = (time.time() - start_time) * 1000
            self.api_call_times.append(execution_time)
            
            # Limit history size
            if len(self.api_call_times) > 100:
                self.api_call_times = self.api_call_times[-50:]
            
            # Check that data is not None before accessing methods
            if data is None:
                self.logger.error(f"No data received from API for {endpoint}")
                return None
                
            if data.get('retCode') == 0:
                return data
            else:
                self.logger.error(f"API error: {data.get('retMsg', 'Unknown error')}")
                return None
                
        except asyncio.TimeoutError:
            self.logger.error(f"Request timeout for {endpoint}")
            return None
        except Exception as e:
            self.logger.error(f"Request error for {endpoint}: {e}")
            return None
    
    async def get_current_price(self) -> Optional[float]:
        """Get current price (with caching)"""
        current_time = time.time()
        
        # Cache for 500ms to avoid unnecessary requests
        if (self.price_cache and 
            current_time - self.price_cache_time < 0.5):
            return self.price_cache
        
        response = await self._make_request("GET", "/v5/market/tickers", {
            "category": config.CATEGORY,
            "symbol": config.SYMBOL
        })
        
        if response and response.get('result', {}).get('list'):
            price = float(response['result']['list'][0]['lastPrice'])
            self.price_cache = price
            self.price_cache_time = current_time
            return price
        
        return None
    
    async def get_position(self) -> float:
        """Get current position (with caching)"""
        current_time = time.time()
        
        # Cache for 1 second
        if (self.position_cache and 
            current_time - self.position_cache_time < 1.0):
            return self.position_cache
        
        response = await self._make_request("GET", "/v5/position/list", {
            "category": config.CATEGORY,
            "symbol": config.SYMBOL
        })
        
        if response and response.get('result', {}).get('list'):
            position_data = response['result']['list'][0]
            size = float(position_data.get('size', 0))
            side = position_data.get('side', '')
            
            # Consider position direction
            if side == 'Sell':
                size = -size
            
            self.position_cache = size
            self.position_cache_time = current_time
            return size
        
        return 0.0
    
    async def get_market_data(self) -> Optional[MarketData]:
        """Get complete market data"""
        response = await self._make_request("GET", "/v5/market/tickers", {
            "category": config.CATEGORY,
            "symbol": config.SYMBOL
        })
        
        if response and response.get('result', {}).get('list'):
            ticker = response['result']['list'][0]
            return MarketData(
                price=float(ticker['lastPrice']),
                bid_price=float(ticker.get('bid1Price', ticker['lastPrice'])),
                ask_price=float(ticker.get('ask1Price', ticker['lastPrice'])),
                timestamp=time.time()
            )
        
        return None
    
    async def place_order(self, side: str, price: float, size: float) -> Optional[str]:
        """Place order"""
        # Parameter validation
        if size < config.MIN_ORDER_SIZE or size > config.MAX_ORDER_SIZE:
            self.logger.warning(f"Invalid order size: {size}")
            return None
        
        # Price and size rounding
        price = round(price / config.TICK_SIZE) * config.TICK_SIZE
        size = round(size, config.SIZE_PRECISION)
        
        params = {
            "category": config.CATEGORY,
            "symbol": config.SYMBOL,
            "side": side,
            "orderType": "Limit",
            "qty": str(size),
            "price": str(price),
            "timeInForce": "PostOnly"  # Only maker orders
        }
        
        response = await self._make_request("POST", "/v5/order/create", params)
        
        if response and response.get('result'):
            order_id = response['result'].get('orderId')
            self.logger.debug(f"Order placed: {side} {size}@{price} -> {order_id}")
            return order_id
        
        return None
    
    async def cancel_all_orders(self) -> bool:
        """Cancel all active orders"""
        response = await self._make_request("POST", "/v5/order/cancel-all", {
            "category": config.CATEGORY,
            "symbol": config.SYMBOL
        })
        
        if response:
            cancelled_count = len(response.get('result', {}).get('list', []))
            if cancelled_count > 0:
                self.logger.debug(f"Cancelled {cancelled_count} orders")
            return True
        
        return False
    
    async def validate_timestamp(self) -> bool:
        """Check time synchronization with server"""
        # No signature needed for public endpoint
        if not self.session:
            self.logger.error("Session not initialized. Use async context manager.")
            return False
            
        try:
            url = f"{self.base_url}/v5/market/time"
            
            async with self.session.get(url) as response:
                data = await response.json()
                
            if data.get('retCode') == 0 and data.get('result'):
                server_time = int(data['result']['timeSecond'])
                local_time = int(time.time())
                time_diff = abs(server_time - local_time) * 1000
                
                if time_diff > config.TIMESTAMP_TOLERANCE_MS:
                    self.logger.error(f"Server time difference: {time_diff}ms")
                    return False
                
                self.logger.info(f"Time synchronized, difference: {time_diff}ms")
                return True
        except Exception as e:
            self.logger.error(f"Time validation error: {e}")
        
        return False
    
    async def get_parallel_market_data(self) -> Tuple[Optional[float], float]:
        """
        Parallel price and position retrieval
        Main optimization for MFT strategies
        """
        start_time = time.time()
        
        # Launch requests in parallel
        price_task = asyncio.create_task(self.get_current_price())
        position_task = asyncio.create_task(self.get_position())
        
        # Wait for both requests to complete
        price, position = await asyncio.gather(price_task, position_task, return_exceptions=True)
        
        # Exception handling
        if isinstance(price, Exception):
            self.logger.error(f"Price fetch error: {price}")
            price = None
        
        if isinstance(position, Exception):
            self.logger.error(f"Position fetch error: {position}")
            position = 0.0
        
        self.parallel_calls_count += 1
        execution_time = (time.time() - start_time) * 1000
        
        self.logger.debug(f"Parallel fetch completed in {execution_time:.1f}ms")
        
        return price, position
    
    async def place_orders_parallel(self, bid_price: float, bid_size: float, 
                                   ask_price: float, ask_size: float) -> Tuple[Optional[str], Optional[str]]:
        """
        Параллельное размещение bid и ask ордеров с возвратом реальных order_id
        """
        # Сначала отменяем старые ордера
        await self.cancel_all_orders()
        
        # Небольшая задержка для обработки отмены
        await asyncio.sleep(0.1)
        
        # Размещаем ордера параллельно
        bid_task = asyncio.create_task(self.place_order("Buy", bid_price, bid_size))
        ask_task = asyncio.create_task(self.place_order("Sell", ask_price, ask_size))
        
        bid_order_id, ask_order_id = await asyncio.gather(bid_task, ask_task, return_exceptions=True)
        
        # Exception handling
        if isinstance(bid_order_id, Exception):
            self.logger.error(f"Bid order error: {bid_order_id}")
            bid_order_id = None
        
        if isinstance(ask_order_id, Exception):
            self.logger.error(f"Ask order error: {ask_order_id}")
            ask_order_id = None
        
        return bid_order_id, ask_order_id
    
    def get_performance_stats(self) -> Dict:
        """Получить статистику производительности"""
        if not self.api_call_times:
            return {}
        
        avg_time = sum(self.api_call_times) / len(self.api_call_times)
        max_time = max(self.api_call_times)
        min_time = min(self.api_call_times)
        
        return {
            'avg_api_time_ms': round(avg_time, 1),
            'max_api_time_ms': round(max_time, 1),
            'min_api_time_ms': round(min_time, 1),
            'parallel_calls': self.parallel_calls_count,
            'total_api_calls': len(self.api_call_times)
        }
    
    # ============================================================================
    # МЕТОДЫ ДЛЯ ТОЧНОГО PnL РАСЧЕТА
    # ============================================================================
    
    async def get_execution_history(self, symbol: str, limit: int = 50, 
                                   start_time: int = None, end_time: int = None) -> List[Dict]:
        """
        Получение истории исполнений через /v5/execution/list
        Возвращает реальные данные о сделках с комиссиями
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = await self._make_request("GET", "/v5/execution/list", params)
            
            if response and response.get('retCode') == 0:
                executions = response.get('result', {}).get('list', [])
                self.logger.debug(f"Retrieved {len(executions)} executions for {symbol}")
                return executions
            else:
                error_msg = response.get('retMsg', 'Unknown error') if response else 'No response'
                self.logger.warning(f"Failed to get execution history: {error_msg}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting execution history: {e}")
            return []

    async def get_trade_details(self, exec_id: str, symbol: str) -> Dict:
        """Получение детальной информации о конкретной сделке"""
        try:
            # Получаем последние исполнения
            executions = await self.get_execution_history(symbol, limit=200)
            
            for execution in executions:
                if execution.get('execId') == exec_id:
                    return {
                        'exec_id': execution.get('execId'),
                        'order_id': execution.get('orderId'),
                        'side': execution.get('side'),
                        'size': float(execution.get('execQty', 0)),
                        'price': float(execution.get('execPrice', 0)),
                        'exec_time': int(execution.get('execTime', 0)),
                        'exec_fee': float(execution.get('execFee', 0)),
                        'fee_rate': float(execution.get('feeRate', 0)),
                        'exec_type': execution.get('execType'),  # Trade, AdlTrade, Funding, etc.
                        'exec_value': float(execution.get('execValue', 0)),
                        'is_maker': execution.get('isMaker', False)
                    }
            
            self.logger.warning(f"Trade details not found for exec_id: {exec_id}")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting trade details for {exec_id}: {e}")
            return {}
    
    async def get_recent_executions_batch(self, symbol: str, lookback_hours: int = 24) -> List[Dict]:
        """
        Получение пакета последних исполнений за указанный период
        Оптимизировано для кэширования
        """
        try:
            current_time = int(time.time() * 1000)
            start_time = current_time - (lookback_hours * 60 * 60 * 1000)
            
            all_executions = []
            remaining_limit = config.EXECUTION_FETCH_LIMIT
            
            # Получаем исполнения пакетами
            while remaining_limit > 0:
                batch_limit = min(50, remaining_limit)  # Bybit лимит 50 за запрос
                
                executions = await self.get_execution_history(
                    symbol=symbol,
                    limit=batch_limit,
                    start_time=start_time,
                    end_time=current_time
                )
                
                if not executions:
                    break
                    
                all_executions.extend(executions)
                remaining_limit -= len(executions)
                
                # Обновляем end_time для следующего пакета (идем назад по времени)
                if executions:
                    last_exec_time = int(executions[-1].get('execTime', current_time))
                    current_time = last_exec_time - 1  # -1мс чтобы избежать дубликатов
                
                # Если получили меньше чем лимит, значит достигли конца
                if len(executions) < batch_limit:
                    break
            
            self.logger.info(f"Retrieved {len(all_executions)} total executions for {symbol} "
                           f"(last {lookback_hours}h)")
            return all_executions
            
        except Exception as e:
            self.logger.error(f"Error getting recent executions batch: {e}")
            return []