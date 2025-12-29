# ---------------------------------------------------------
# Stock Detail API
# ---------------------------------------------------------
@app.get("/stockdetail/{symbol}")
def get_stock_detail(symbol: str):
    """
    Returns full stock intelligence for Stock Detail screen.
    Safe, cached, cost-efficient.
    """

    symbol = symbol.upper().strip()

    if not symbol.isalpha():
        return {
            "status": "error",
            "error": "Invalid symbol"
        }

    try:
        stock = bootstrap_stock(symbol)

        return {
            "status": "ok",
            "symbol": symbol,
            "data": stock,
        }

    except Exception as e:
        return {
            "status": "error",
            "symbol": symbol,
            "error": str(e),
        }