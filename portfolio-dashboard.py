import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Zerodha Trading Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better Styling ---
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
        --background-light: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, #MainMenu, header, footer {
        visibility: hidden;
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0.25rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    
    /* Status indicators */
    .status-connected {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
    }
    
    .status-monitoring {
        background: linear-gradient(135deg, #3498db, #2980b9);
    }
    
    .status-stopped {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
    }
    
    .status-error {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background-color: #d4edda;
        border-color: #2ecc71;
        color: #155724;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border-color: #e74c3c;
        color: #721c24;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border-color: #f39c12;
        color: #856404;
    }
    
    /* Button enhancements */
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Log container styling */
    .log-container {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def pnl_color(val):
    """Adds color to the P&L column based on its value."""
    if pd.isna(val):
        return ''
    color = '#e74c3c' if val < 0 else '#2ecc71'
    return f'color: {color}; font-weight: bold;'

def add_log(message, log_type="info"):
    """Adds a timestamped message to the log with type."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    emoji_map = {
        "info": "ℹ️",
        "success": "✅", 
        "error": "❌",
        "warning": "⚠️",
        "trigger": "🚨"
    }
    emoji = emoji_map.get(log_type, "ℹ️")
    st.session_state.logs.insert(0, f"{timestamp} {emoji} {message}")
    if len(st.session_state.logs) > 100:  # Limit log size
        st.session_state.logs = st.session_state.logs[:100]

def is_option_symbol(tradingsymbol):
    """Check if the trading symbol is an option."""
    return ('CE' in tradingsymbol or 'PE' in tradingsymbol) and any(char.isdigit() for char in tradingsymbol)

def get_optimal_exit_price(kite, instrument, ltp, transaction_type, tradingsymbol):
    """Get optimal exit price based on market depth."""
    try:
        # Get market depth
        quote = kite.quote(instrument)
        instrument_quote = quote.get(instrument, {})
        depth = instrument_quote.get('depth', {})
        
        if transaction_type == kite.TRANSACTION_TYPE_SELL:
            # For selling, try to get the best bid price
            buy_orders = depth.get('buy', [])
            if buy_orders:
                best_bid = buy_orders[0].get('price', ltp)
                # Use bid price but not more than 2% below LTP for quick execution
                return max(best_bid, ltp * 0.98)
            else:
                return ltp * 0.98  # 2% below LTP if no bids
        else:
            # For buying (covering short), try to get the best ask price
            sell_orders = depth.get('sell', [])
            if sell_orders:
                best_ask = sell_orders[0].get('price', ltp)
                # Use ask price but not more than 2% above LTP
                return min(best_ask, ltp * 1.02)
            else:
                return ltp * 1.02  # 2% above LTP if no asks
                
    except Exception as e:
        add_log(f"Error getting market depth for {tradingsymbol}: {e}", "error")
        # Fallback pricing
        if transaction_type == kite.TRANSACTION_TYPE_SELL:
            return ltp * 0.98
        else:
            return ltp * 1.02
    """Creates a P&L visualization chart."""
    if not position_details:
        return None
    
    df = pd.DataFrame(position_details)
    df['PnL_numeric'] = df['P&L']
    
    fig = go.Figure()
    
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df['PnL_numeric']]
    
    fig.add_trace(go.Bar(
        x=df['Symbol'],
        y=df['PnL_numeric'],
        marker_color=colors,
        text=[f'₹{x:,.0f}' for x in df['PnL_numeric']],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>P&L: ₹%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Position-wise P&L",
        xaxis_title="Symbols",
        yaxis_title="P&L (₹)",
        template="plotly_white",
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# --- Session State Initialization ---
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'total_pnl_history' not in st.session_state:
    st.session_state.total_pnl_history = []
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# --- Header ---
st.markdown("""
# 🚀 Zerodha Trading Copilot
### Advanced Portfolio Monitoring & Risk Management
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("## 🔐 Authentication")
    
    with st.expander("API Credentials", expanded=not st.session_state.monitoring):
        api_key = st.text_input("API Key", type="password", help="Your Zerodha API Key")
        api_secret = st.text_input("API Secret", type="password", help="Your Zerodha API Secret")
        access_token = st.text_input("Access Token", type="password", help="Your Access Token")
    
    st.markdown("## ⚙️ Risk Management")
    
    max_loss_per_trade = st.number_input(
        "Stop Loss per Trade (₹)",
        min_value=100,
        max_value=100000,
        value=5000,
        step=500,
        help="Maximum loss allowed per position before auto-exit"
    )
    
    max_portfolio_loss = st.number_input(
        "Portfolio Stop Loss (₹)",
        min_value=1000,
        max_value=500000,
        value=20000,
        step=1000,
        help="Maximum total portfolio loss before stopping all trades"
    )
    
    st.markdown("## 🔄 Monitoring Settings")
    
    poll_interval = st.slider(
        "Refresh Interval (seconds)", 
        min_value=1, 
        max_value=30, 
        value=3,
        help="How often to check positions"
    )
    
    enable_sound_alerts = st.checkbox("🔊 Sound Alerts", value=True)
    
    st.markdown("## 📊 Display Options")
    
    show_chart = st.checkbox("📈 Show P&L Chart", value=True)
    show_detailed_logs = st.checkbox("📜 Detailed Logs", value=True)
    
    st.markdown("---")
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button(
            "🚀 Start", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.monitoring
        )
    
    with col2:
        stop_button = st.button(
            "⏹️ Stop", 
            type="secondary", 
            use_container_width=True,
            disabled=not st.session_state.monitoring
        )

# --- Main Dashboard ---

# Status indicators
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.session_state.monitoring:
        status_class = "status-monitoring"
        status_text = "MONITORING"
        status_icon = "🟢"
    else:
        status_class = "status-stopped"
        status_text = "STOPPED"
        status_icon = "🔴"
    
    st.markdown(f"""
    <div class="metric-card {status_class}">
        <h3>{status_icon}</h3>
        <p>{status_text}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    connection_status = "CONNECTED" if st.session_state.kite else "DISCONNECTED"
    connection_icon = "🔗" if st.session_state.kite else "🔴"
    connection_class = "status-connected" if st.session_state.kite else "status-error"
    
    st.markdown(f"""
    <div class="metric-card {connection_class}">
        <h3>{connection_icon}</h3>
        <p>{connection_status}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    last_refresh = st.session_state.last_refresh.strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="metric-card">
        <h3>🕐</h3>
        <p>Last Update: {last_refresh}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if st.session_state.user_profile:
        user_name = st.session_state.user_profile.get('user_name', 'User')
        st.markdown(f"""
        <div class="metric-card status-connected">
            <h3>👤</h3>
            <p>{user_name}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card status-error">
            <h3>👤</h3>
            <p>Not Logged In</p>
        </div>
        """, unsafe_allow_html=True)

# --- Authentication Logic ---
if start_button:
    if not all([api_key, api_secret, access_token]):
        st.error("🔒 Please provide all credentials in the sidebar.")
    else:
        with st.spinner("Connecting to Zerodha..."):
            try:
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)
                
                # Test connection
                profile = kite.profile()
                st.session_state.kite = kite
                st.session_state.user_profile = profile
                st.session_state.monitoring = True
                
                add_log(f"Connected successfully as {profile.get('user_name', 'User')}", "success")
                st.success(f"✅ Connected! Welcome {profile.get('user_name', 'User')}")
                
            except Exception as e:
                add_log(f"Connection failed: {str(e)}", "error")
                st.error(f"❌ Connection failed: {str(e)}")
                st.session_state.monitoring = False

if stop_button:
    st.session_state.monitoring = False
    add_log("Monitoring stopped by user", "warning")
    st.warning("⏹️ Monitoring stopped")

# --- Main Dashboard Content ---
st.markdown("---")

if st.session_state.monitoring and st.session_state.kite:
    kite = st.session_state.kite
    
    try:
        # Fetch data
        positions = kite.positions().get('net', [])
        open_positions = [p for p in positions if p['quantity'] != 0]
        
        # Update refresh time
        st.session_state.last_refresh = datetime.now()
        
        if not open_positions:
            st.info("📊 No open positions to monitor")
            
        else:
            # Fetch LTP data
            instrument_list = [f"{p['exchange']}:{p['tradingsymbol']}" for p in open_positions]
            ltp_data = kite.ltp(instrument_list)
            
            total_pnl = 0
            position_details = []
            alerts = []

            for pos in open_positions:
                tradingsymbol = pos['tradingsymbol']
                instrument = f"{pos['exchange']}:{tradingsymbol}"
                ltp = ltp_data.get(instrument, {}).get('last_price', 0)
                
                pnl = (ltp - pos['average_price']) * pos['quantity']
                total_pnl += pnl
                
                # Calculate percentage change
                pnl_percent = (pnl / (pos['average_price'] * abs(pos['quantity']))) * 100
                
                position_details.append({
                    "Symbol": tradingsymbol,
                    "Quantity": pos['quantity'],
                    "Avg. Price": pos['average_price'],
                    "LTP": ltp,
                    "P&L": pnl,
                    "P&L %": pnl_percent,
                    "Product": pos['product'],
                    "Exchange": pos['exchange']
                })
                
                # Check stop-loss triggers
                if pnl <= -abs(max_loss_per_trade):
                    alerts.append(f"🚨 Stop-loss triggered for {tradingsymbol} (₹{pnl:,.2f})")
                    add_log(f"Stop-loss triggered for {tradingsymbol} at P&L: ₹{pnl:.2f}", "trigger")
                    
                    # Auto-exit logic with intelligent order placement
                    exit_transaction_type = kite.TRANSACTION_TYPE_SELL if pos['quantity'] > 0 else kite.TRANSACTION_TYPE_BUY
                    
                    try:
                        # Check if it's an option
                        if is_option_symbol(tradingsymbol):
                            # For options, always use limit orders
                            limit_price = get_optimal_exit_price(kite, instrument, ltp, exit_transaction_type, tradingsymbol)
                            
                            order_id = kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=pos['exchange'],
                                tradingsymbol=tradingsymbol,
                                transaction_type=exit_transaction_type,
                                quantity=abs(pos['quantity']),
                                product=pos['product'],
                                order_type=kite.ORDER_TYPE_LIMIT,
                                price=round(limit_price, 2)
                            )
                            add_log(f"Options exit order: {tradingsymbol} at ₹{limit_price:.2f} (Order ID: {order_id})", "success")
                        
                        else:
                            # For stocks, try market order first, fallback to limit
                            try:
                                order_id = kite.place_order(
                                    variety=kite.VARIETY_REGULAR,
                                    exchange=pos['exchange'],
                                    tradingsymbol=tradingsymbol,
                                    transaction_type=exit_transaction_type,
                                    quantity=abs(pos['quantity']),
                                    product=pos['product'],
                                    order_type=kite.ORDER_TYPE_MARKET
                                )
                                add_log(f"Stock exit order: {tradingsymbol} at market price (Order ID: {order_id})", "success")
                            except Exception as market_error:
                                # Fallback to limit order
                                limit_price = get_optimal_exit_price(kite, instrument, ltp, exit_transaction_type, tradingsymbol)
                                order_id = kite.place_order(
                                    variety=kite.VARIETY_REGULAR,
                                    exchange=pos['exchange'],
                                    tradingsymbol=tradingsymbol,
                                    transaction_type=exit_transaction_type,
                                    quantity=abs(pos['quantity']),
                                    product=pos['product'],
                                    order_type=kite.ORDER_TYPE_LIMIT,
                                    price=round(limit_price, 2)
                                )
                                add_log(f"Stock exit order (limit): {tradingsymbol} at ₹{limit_price:.2f} (Order ID: {order_id})", "success")
                    
                    except Exception as e:
                        add_log(f"Failed to place exit order for {tradingsymbol}: {e}", "error")
                        # Try a more aggressive limit order as last resort
                        try:
                            aggressive_price = ltp * 0.95 if exit_transaction_type == kite.TRANSACTION_TYPE_SELL else ltp * 1.05
                            order_id = kite.place_order(
                                variety=kite.VARIETY_REGULAR,
                                exchange=pos['exchange'],
                                tradingsymbol=tradingsymbol,
                                transaction_type=exit_transaction_type,
                                quantity=abs(pos['quantity']),
                                product=pos['product'],
                                order_type=kite.ORDER_TYPE_LIMIT,
                                price=round(aggressive_price, 2)
                            )
                            add_log(f"Emergency exit order: {tradingsymbol} at ₹{aggressive_price:.2f} (Order ID: {order_id})", "warning")
                        except Exception as final_error:
                            add_log(f"All exit attempts failed for {tradingsymbol}: {final_error}", "error")
            
            # Store P&L history for trending
            st.session_state.total_pnl_history.append({
                'timestamp': datetime.now(),
                'total_pnl': total_pnl
            })
            
            # Keep only last 100 data points
            if len(st.session_state.total_pnl_history) > 100:
                st.session_state.total_pnl_history = st.session_state.total_pnl_history[-100:]
            
            # Portfolio-level stop loss
            if total_pnl <= -abs(max_portfolio_loss):
                st.error(f"🚨 PORTFOLIO STOP LOSS TRIGGERED! Total P&L: ₹{total_pnl:,.2f}")
                add_log(f"Portfolio stop-loss triggered at ₹{total_pnl:.2f}", "trigger")
                # Here you could add logic to close all positions
            
            # Display alerts
            for alert in alerts:
                st.error(alert)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pnl_color_class = "success" if total_pnl >= 0 else "danger"
                st.markdown(f"""
                <div class="alert-box alert-{pnl_color_class}">
                    <h3 style="margin:0;">₹{total_pnl:,.2f}</h3>
                    <p style="margin:0;">Total P&L</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="alert-box alert-success">
                    <h3 style="margin:0;">{len(open_positions)}</h3>
                    <p style="margin:0;">Open Positions</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                winning_positions = sum(1 for p in position_details if p['P&L'] > 0)
                st.markdown(f"""
                <div class="alert-box alert-success">
                    <h3 style="margin:0;">{winning_positions}</h3>
                    <p style="margin:0;">Winning</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                losing_positions = sum(1 for p in position_details if p['P&L'] < 0)
                st.markdown(f"""
                <div class="alert-box alert-danger">
                    <h3 style="margin:0;">{losing_positions}</h3>
                    <p style="margin:0;">Losing</p>
                </div>
                """, unsafe_allow_html=True)
            
            # P&L Chart
            if show_chart and position_details:
                st.plotly_chart(create_pnl_chart(position_details), use_container_width=True)
            
            # Positions table
            st.subheader("📊 Live Positions")
            df = pd.DataFrame(position_details)
            
            # Format the dataframe for display
            display_df = df.copy()
            display_df['Avg. Price'] = display_df['Avg. Price'].apply(lambda x: f"₹{x:.2f}")
            display_df['LTP'] = display_df['LTP'].apply(lambda x: f"₹{x:.2f}")
            display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:.2f}%")
            
            # Style the dataframe
            styled_df = display_df.style.format({
                "P&L": "₹{:,.2f}"
            }).applymap(pnl_color, subset=['P&L']).applymap(
                lambda x: 'font-weight: bold' if 'P&L' in str(x) else '', 
                subset=['P&L %']
            )
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
        add_log(f"Error: {str(e)}", "error")
        st.session_state.monitoring = False

elif not st.session_state.monitoring:
    st.info("🔌 Connect and start monitoring to see your positions")

# --- Event Logs ---
if show_detailed_logs and st.session_state.logs:
    st.subheader("📜 Event Logs")
    
    log_text = "\n".join(st.session_state.logs[:20])  # Show last 20 logs
    st.text_area("", value=log_text, height=200, disabled=True)

# --- Auto-refresh ---
if st.session_state.monitoring:
    time.sleep(poll_interval)
    st.rerun()