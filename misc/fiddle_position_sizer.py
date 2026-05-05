def calculate_position_size(account_balance:float, entry_price:float, stop_loss_price:float, risk_pct:float, min_position:float=0.1) -> dict:
    #check
    if account_balance <= 0:
        raise ValueError("account_balance must be positive")
    if entry_price == stop_loss_price:
        raise ValueError("entry_price can't equal stop_loss_price")
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("entry_price or stop_loss_price must be positive")
    if risk_pct <= 0 or risk_pct >= 0.1:
        raise ValueError("risk_pct must be between 0.00 and 0.10")
    
    #calculations
    max_risk = account_balance * risk_pct
    risk_per_unit = abs(entry_price - stop_loss_price)
    units_to_trade = max_risk / risk_per_unit
    position_size = units_to_trade * entry_price

    #check
    if position_size < min_position:
        raise ValueError("position_size is too small")
    
    #direction
    direction = "LONG" if entry_price > stop_loss_price else "SHORT"

    return{
        "Account Balance": account_balance,
        "Entry": entry_price,
        "Stop": stop_loss_price,
        "Direction": direction,
        "Risk_%": risk_pct,
        "Max Risk Allowed": round(max_risk, 2),
        "Risk per Unit": risk_per_unit,
        "Units to Trade": units_to_trade,
        "Position Size $": round(position_size, 2)
    }

if __name__ == "__main__":
    result = calculate_position_size(1000, 100, 99, 0.02)
    for key, value in result.items():
        print(f"{key:20} : {value}")

