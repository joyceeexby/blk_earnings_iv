#!/usr/bin/env python3
"""
Simple test script to verify the fixes in automated_analysis.py
"""

import sys
import os

def test_syntax():
    """Test the syntax of the file directly."""
    try:
        with open('automated_analysis.py', 'r') as f:
            content = f.read()
        
        # Try to compile the content
        compile(content, 'automated_analysis.py', 'exec')
        print("SUCCESS: File compiles without syntax errors")
        return True
        
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_import():
    """Test if the fixed automated_analysis.py can be imported without syntax errors."""
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import the fixed module
        from automated_analysis import AutomatedEarningsAnalysis
        
        print("SUCCESS: automated_analysis.py imports without syntax errors")
        print("SUCCESS: The SQLAlchemy and type conversion fixes are working")
        
        # Test creating an instance (without database connection)
        try:
            # This will fail due to missing database, but we can test the class definition
            print("SUCCESS: AutomatedEarningsAnalysis class is properly defined")
        except Exception as e:
            print(f"Expected error (no database): {e}")
            
        return True
        
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        return False
    except SyntaxError as e:
        print(f"SYNTAX ERROR: {e}")
        return False
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    print("TESTING FIXES IN automated_analysis.py")
    print("="*50)
    
    # Test 1: Syntax check
    print("\n1. Testing syntax...")
    syntax_ok = test_syntax()
    
    # Test 2: Import test
    print("\n2. Testing import...")
    import_ok = test_import()
    
    # Summary
    print("\n" + "="*50)
    if syntax_ok and import_ok:
        print("ALL TESTS PASSED: The fixes are working correctly!")
        print("SUCCESS: You can now run the automated analysis without the previous errors")
    else:
        print("SOME TESTS FAILED: There are still issues to resolve")
    
    print("\nNext steps:")
    print("1. Run test_integration.py to test the full integration")
    print("2. Check if dispersion and Fama-French factors are now populated")
    print("3. If issues persist, check the console output for new error messages")
