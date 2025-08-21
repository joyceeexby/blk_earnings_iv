# Configuration Examples for Option Surface Features Generation

## 🎯 How to Configure Date Ranges

Edit the configurable parameters in `generate_option_features_2005_2023.py`:

```python
# =============================================================================
# CONFIGURABLE PARAMETERS - MODIFY THESE TO SET YOUR DATE RANGE
# =============================================================================

# Set your desired start and end dates here
START_YEAR = 2012
START_QUARTER = 4  # 1=Q1, 2=Q2, 3=Q3, 4=Q4
END_YEAR = 2023
END_QUARTER = 4    # 1=Q1, 2=Q2, 3=Q3, 4=Q4

# =============================================================================
# END OF CONFIGURABLE PARAMETERS
# =============================================================================
```

## 📅 Configuration Examples

### **Example 1: Generate Q4 2012 to Q4 2023 (Current Default)**
```python
START_YEAR = 2012
START_QUARTER = 4  # Q4
END_YEAR = 2023
END_QUARTER = 4    # Q4
```
**Result**: Processes 45 seasons (Q4 2012 through Q4 2023)

### **Example 2: Generate Full Year 2013**
```python
START_YEAR = 2013
START_QUARTER = 1  # Q1
END_YEAR = 2013
END_QUARTER = 4    # Q4
```
**Result**: Processes 4 seasons (Q1-Q4 2013)

### **Example 3: Generate Q2 2015 to Q1 2016**
```python
START_YEAR = 2015
START_QUARTER = 2  # Q2
END_YEAR = 2016
END_QUARTER = 1    # Q1
```
**Result**: Processes 4 seasons (Q2-Q4 2015, Q1 2016)

### **Example 4: Generate Single Quarter (Q3 2020)**
```python
START_YEAR = 2020
START_QUARTER = 3  # Q3
END_YEAR = 2020
END_QUARTER = 3    # Q3
```
**Result**: Processes 1 season (Q3 2020 only)

### **Example 5: Generate Recent Years (2018-2023)**
```python
START_YEAR = 2018
START_QUARTER = 1  # Q1
END_YEAR = 2023
END_QUARTER = 4    # Q4
```
**Result**: Processes 24 seasons (Q1 2018 through Q4 2023)

## 🔄 Smart File Skipping

The script automatically:
- ✅ **Checks** if CSV files already exist in `data_files/`
- ⏭️ **Skips** seasons that have already been generated
- 📊 **Reports** how many files were skipped vs. newly generated
- 🚀 **Continues** from where it left off

## 📋 Usage

1. **Set your desired date range** in the configurable parameters
2. **Run the script**: `python run_option_features_generation.py`
3. **Monitor progress** - existing files will be automatically skipped
4. **Check results** in `data_files/` folder

## 🎯 Tips

- **Start small**: Test with a single quarter first
- **Resume anytime**: The script will skip already generated files
- **Monitor disk space**: Large date ranges generate many CSV files
- **Check logs**: The script shows detailed progress and skipping information
