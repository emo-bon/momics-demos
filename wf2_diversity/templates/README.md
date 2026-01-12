# WF2 Diversity Templates

This folder contains reusable Panel templates for the diversity analysis notebooks.

## Available Templates

### Loading Tab (`loading_tab.py`)

A simple data loading interface for viewing sample information.

#### Features
- File selection from available CSV files
- Sample count indicator
- Status feedback

#### Usage
```python
from templates import create_loading_tab

loading_tab_content, loading_state = create_loading_tab(
    root_folder=root_folder,
    full_metadata=full_metadata,
    mgf_parquet_dfs=mgf_parquet_dfs,
    on_filter_callback=my_callback_function,
)
```

### UDAL Filter Tab (`udal_filter_tab.py`)

A comprehensive UDAL query-based filtering interface for advanced data selection.

#### Features

- **UDAL Query Builder**: Text area for entering standard Pandas query syntax
- **Preset Queries**: Quick filters for common scenarios:
  - Winter samples
  - Deep water (>100m)
  - Surface samples
  - High salinity
  - Recent samples (2023+)
- **Query Validation**: Test queries before applying them
- **Multi-table Filtering**: Select which tables (LSU/SSU) to filter
- **Query History**: Track all applied queries with results
- **Real-time Indicators**:
  - Original row count
  - Filtered row count
  - Reduction percentage
- **Data Preview**: View first 50 rows of filtered data
- **Column Information**: Display available columns for querying

#### Usage

```python
from templates import create_udal_filter_tab

# After loading metadata and data
udal_filter_tab_content, udal_filter_state = create_udal_filter_tab(
    full_metadata=full_metadata,
    mgf_parquet_dfs=mgf_parquet_dfs,
    on_filter_callback=my_callback_function,
)

# Access filtered data via udal_filter_state
filtered_metadata = udal_filter_state['filtered_metadata']
filtered_tables = udal_filter_state['filtered_tables']
```

#### UDAL Query Examples

```python
# Simple equality
season == "winter"

# Numerical comparison
depth > 100

# Multiple values
season.isin(["winter", "spring"])

# Combined conditions
(depth > 50) & (season == "winter")

# String matching
sample_id.str.contains("EMOBON")

# Column names with spaces (use backticks)
`observatory name` == "Oslo"

# Complex queries
(depth > 100) | (sea_surface_temperature < 10) & (season != "summer")
```

#### Query Operators
- `==` - Equal to
- `!=` - Not equal to
- `>`, `<`, `>=`, `<=` - Numerical comparisons
- `&` - AND (both conditions must be true)
- `|` - OR (at least one condition must be true)
- `~` - NOT (negates the condition)
- `.isin([...])` - Check if value is in list
- `.str.contains("...")` - String pattern matching
- `.str.startswith("...")` - String starts with
- `.isna()`, `.notna()` - Check for missing values

#### Callback Function

```python
def on_udal_filter_applied(state):
    """Handle UDAL filter application."""
    if state['filtered_metadata'] is not None:
        print(f"Filtered rows: {len(state['filtered_metadata'])}")
    if state['filtered_tables'] is not None:
        print(f"Filtered tables: {list(state['filtered_tables'].keys())}")
```

#### State Dictionary

- `original_metadata`: Original unfiltered metadata DataFrame
- `original_tables`: Original unfiltered data tables dictionary
- `filtered_metadata`: Filtered metadata DataFrame (None if no filter applied)
- `filtered_tables`: Dictionary of filtered data tables (None if no filter applied)
- `query_history`: List of all applied queries with their results

## Integration Example

See [diversities_panel.ipynb](../diversities_panel.ipynb) for complete integration examples.

## Design Principles

- **Modularity**: Each template should be self-contained and reusable
- **Flexibility**: Accept parameters and callbacks for customization
- **Consistency**: Follow Panel UI patterns used in existing notebooks
- **Documentation**: Include docstrings and usage examples
- **Error Handling**: Gracefully handle missing data or configuration errors
