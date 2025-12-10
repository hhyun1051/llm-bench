"""Function calling에 사용될 tool 함수들"""


def get_weather(city: str, unit: str = "celsius") -> str:
    """Get weather for a given city.

    Args:
        city: City name
        unit: Temperature unit (celsius or fahrenheit)
    """
    temps = {
        "서울": 22, "seoul": 22,
        "부산": 24, "busan": 24,
        "san francisco": 18, "sf": 18,
        "new york": 15, "ny": 15,
        "london": 12,
        "tokyo": 20, "도쿄": 20
    }
    temp = temps.get(city.lower(), 20)

    if unit == "fahrenheit":
        temp = int(temp * 9/5 + 32)
        return f"The weather in {city} is {temp}°F, sunny"

    return f"The weather in {city} is {temp}°C, sunny"


def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic arithmetic operations.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
    """
    ops = {
        'add': a + b,
        'subtract': a - b,
        'multiply': a * b,
        'divide': a / b if b != 0 else "Error: Division by zero"
    }
    result = ops.get(operation, "Invalid operation")
    return result


def search_database(query: str, table: str = "users") -> str:
    """Search database for information.

    Args:
        query: Search query
        table: Database table name
    """
    # Mock database search
    mock_results = {
        "users": 5,
        "products": 12,
        "orders": 8,
        "customers": 15
    }
    count = mock_results.get(table, 3)
    return f"Found {count} results in '{table}' table for query: '{query}'"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content
    """
    return f"Email sent successfully to {to} with subject '{subject}'"


def get_current_time(timezone: str = "UTC") -> str:
    """Get current time in specified timezone.

    Args:
        timezone: Timezone name (e.g., 'UTC', 'Asia/Seoul', 'America/New_York')
    """
    from datetime import datetime
    return f"Current time in {timezone}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency from one to another.

    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., 'USD', 'KRW', 'EUR')
        to_currency: Target currency code
    """
    # Mock exchange rates
    rates = {
        ('USD', 'KRW'): 1300,
        ('KRW', 'USD'): 1/1300,
        ('USD', 'EUR'): 0.85,
        ('EUR', 'USD'): 1/0.85,
        ('USD', 'JPY'): 110,
        ('JPY', 'USD'): 1/110
    }

    rate = rates.get((from_currency, to_currency), 1.0)
    result = amount * rate
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"
