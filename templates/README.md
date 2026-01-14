# Momics-Demos Templates

This folder contains reusable Panel templates that can be used across all workflow notebooks (wf0, wf1, wf2, wf3, wf4, wf5, wfs_extra).

## Purpose

The templates provide common UI components and functionality that multiple workflows need:
- Data filtering interfaces
- Data loading panels
- Visualization controls
- Common widgets and layouts

## Available Templates

### Data Filter Tab (`data_filter_tab.py`)

A comprehensive data filtering interface using Pandas query syntax.

#### Features

- **Query Builder**: Text area for entering standard Pandas query syntax
- **Preset Queries**: Quick filters for common scenarios:
  - Winter samples
  - Deep water (>100m)
  - Surface samples
  - High salinity
  - Recent samples (2022+)
  - Combined filters
- **Query Validation**: Test queries before applying them
- **Multi-table Filtering**: Select which tables to filter
- **Query History**: Track all applied queries with results
- **Real-time Indicators**:
  - Original row count
  - Filtered row count
  - Reduction percentage
- **Column Information**: Collapsible card showing available columns

#### Usage

```python
from templates import create_data_filter_tab

# After loading metadata and data
data_filter_tab_content, data_filter_state = create_data_filter_tab(
    full_metadata=full_metadata,
    mgf_parquet_dfs=mgf_parquet_dfs,
    on_filter_callback=my_callback_function,
)

# Access filtered data via state
filtered_metadata = data_filter_state['filtered_metadata']
filtered_tables = data_filter_state['filtered_tables']
```

#### Query Examples

```python
# Simple equality
season == "Winter"

# Numerical comparison
`sampling depth (m)` > 100

# Multiple values
season.isin(["Winter", "Spring"])

# Combined conditions
(`sampling depth (m)` > 50) & (season == "Winter")

# String matching
sample_id.str.contains("EMOBON")

# Column names with spaces (use backticks)
`observatory name` == "Oslo"
```

#### Callback Function

```python
def on_data_filter_applied(state):
    """Handle filter application."""
    global full_metadata, mgf_parquet_dfs, tables
    
    # Update global variables with filtered data
    full_metadata = state['filtered_metadata'] if state['filtered_metadata'] is not None else state['original_metadata']
    mgf_parquet_dfs = state['filtered_tables'] if state['filtered_tables'] is not None else state['original_tables']
    
    # Trigger plot updates
    filter_trigger.value += 1
```

#### State Dictionary

- `original_metadata`: Original unfiltered metadata DataFrame
- `original_tables`: Original unfiltered data tables dictionary
- `filtered_metadata`: Filtered metadata DataFrame (None if no filter applied)
- `filtered_tables`: Dictionary of filtered data tables (None if no filter applied)
- `query_history`: List of all applied queries with their results

## Integration in Workflows

### Example: wf2_diversity

```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))  # Add repo root to path
from templates import create_data_filter_tab

# Create filter trigger for reactive updates
filter_trigger = pn.widgets.IntInput(value=0, visible=False)

# Define callback
def on_data_filter_applied(state):
    global full_metadata, mgf_parquet_dfs, tables
    full_metadata = state['filtered_metadata'] if state['filtered_metadata'] is not None else state['original_metadata']
    mgf_parquet_dfs = state['filtered_tables'] if state['filtered_tables'] is not None else state['original_tables']
    tables = mgf_parquet_dfs
    filter_trigger.value += 1

# Create the tab
data_filter_tab_content, data_filter_state = create_data_filter_tab(
    full_metadata=full_metadata,
    mgf_parquet_dfs=mgf_parquet_dfs,
    on_filter_callback=on_data_filter_applied,
)

# Add to Panel tabs
tabs = pn.Tabs(
    ('Data Filter', data_filter_tab_content),
    ('Analysis', analysis_content),
    # ... other tabs
)
```

## Design Principles

- **Modularity**: Each template is self-contained and reusable
- **Flexibility**: Accept parameters and callbacks for customization
- **Consistency**: Follow Panel UI patterns used across notebooks
- **Documentation**: Include docstrings and usage examples
- **Error Handling**: Gracefully handle missing data or configuration errors
- **Accessibility**: Templates work across all workflow notebooks

## Adding New Templates

When creating a new template:

1. Create a new Python file in this folder
2. Follow the naming convention: `descriptive_name_tab.py`
3. Include comprehensive docstrings
4. Add it to `__init__.py` for easy importing
5. Update this README with usage examples
6. Test across multiple workflows to ensure compatibility

## Dependencies

Templates use:
- `panel` - For UI components
- `pandas` - For data manipulation
- Standard Python logging

Make sure these are available in your notebook environment.
