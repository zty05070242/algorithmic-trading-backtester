
def calculate_position_size(account_balance:float, entry_price:float, stop_loss_price:float, risk_pct:float, min_position:float=0.1) -> dict:
   
    # Safety: validate any input before any calculation
    if account_balance <= 0:
        raise ValueError("Account balance must be above 0")
    if risk_pct <= 0 or risk_pct > 0.1:
        raise ValueError("Risk percentage must be between 0 and 0.1")
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("Entry and stop loss must be positive")
    if entry_price == stop_loss_price:
        raise ValueError("Entry should not equal to stop loss.")
 
    max_risk = account_balance * risk_pct
    risk_per_unit = abs(entry_price - stop_loss_price)
    units_to_trade = max_risk / risk_per_unit
    position_size = units_to_trade * entry_price

    # Safety: check values after calculation as well.
    if risk_per_unit < 0.01:
        raise ValueError("Entry and stop loss are too close")
    if position_size < min_position:
        raise ValueError(f"Calculated position size {position_size} is smaller than the minimim: {min_position}. Adjust your levels.")

    direction = "LONG" if entry_price > stop_loss_price else "SHORT"


    return{
        "Account Balance": account_balance,
        "Entry Price": entry_price,
        "Stop Loss Price": stop_loss_price,
        "Direction": direction,
        "Risk %": risk_pct,
        "Max Risk Allowed": round(max_risk, 2),
        "Risk per Unit": risk_per_unit,
        "Units to Trade": units_to_trade,
        "Position Size": round(position_size, 2)
        }

# test the code with main guard
if __name__ == "__main__":
    position = calculate_position_size(10000, 100, 99, 0.02)
    for key, value in position.items():
        print(f"{key:15} : {value}")

