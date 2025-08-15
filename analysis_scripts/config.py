# WRDS Database Configuration
# Credentials will be prompted for when the code runs
import getpass

def get_wrds_credentials():
    """
    Prompt user for WRDS credentials interactively.
    """
    print("WRDS Database Access Required")
    print("="*40)
    username = input("Enter your WRDS username: ")
    password = getpass.getpass("Enter your WRDS password: ")
    return username, password

# Analysis Parameters
# Full test configuration (2 years, 45 stocks)
START_DATE = '2022-01-01'
END_DATE = '2023-12-31'
ANALYSIS_DAYS_BEFORE = 30
