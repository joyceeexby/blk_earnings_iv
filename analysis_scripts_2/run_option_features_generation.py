#!/usr/bin/env python3
"""
Execution Script for Option Surface Features Generation 2007-2023
Simple script to run the option features generation with WRDS connection
Modified to start from 2007 (skipping 2005-2006)
"""

import wrds
from generate_option_features_2005_2023 import (
    generate_all_option_features_2005_2023,
    create_summary_report
)

def main():
    """
    Main execution function with WRDS connection.
    """
    print("🚀 STARTING OPTION SURFACE FEATURES GENERATION (2007-2023)")
    print("="*80)
    print("⏭️ Skipping 2005-2006 (already completed)")
    
    try:
        # Connect to WRDS
        print("🔌 Connecting to WRDS...")
        db = wrds.Connection()
        print("✅ WRDS connection established")
        
        # Generate option features for all seasons from 2007-2023
        print("\n🔄 Starting generation process...")
        generated_files = generate_all_option_features_2005_2023(db)
        
        # Create summary report
        if generated_files:
            print("\n📊 Creating summary report...")
            combined_file = create_summary_report(generated_files)
            
            if combined_file:
                print(f"\n🎉 SUCCESS! Combined data saved to: {combined_file}")
            else:
                print("\n⚠️ Summary report creation failed")
        else:
            print("\n❌ No files were generated")
        
        # Close WRDS connection
        db.close()
        print("🔌 WRDS connection closed")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("Please check your WRDS credentials and connection")

if __name__ == "__main__":
    main()
