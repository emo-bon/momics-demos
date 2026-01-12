"""
UDAL Filter Tab Template for Panel Applications
Creates a reusable tab component for filtering data using UDAL queries.
"""

import os
import logging
import pandas as pd
import panel as pn

logger = logging.getLogger(__name__)


def create_udal_filter_tab(
    full_metadata: pd.DataFrame = None,
    mgf_parquet_dfs: dict = None,
    on_filter_callback=None,
):
    """
    Creates a Panel tab for filtering data using UDAL queries.
    
    Args:
        full_metadata (pd.DataFrame, optional): Full metadata dataframe
        mgf_parquet_dfs (dict, optional): Dictionary of parquet dataframes
        on_filter_callback (callable, optional): Callback function when filters are applied
        
    Returns:
        tuple: (pn.Column, state_dict) - Panel column containing the UDAL filter tab UI and state dictionary
    """
    
    # State management
    state = {
        'original_metadata': full_metadata,
        'original_tables': mgf_parquet_dfs,
        'filtered_metadata': None,
        'filtered_tables': None,
        'query_history': [],
    }
    
    # ========== WIDGETS ==========
    
    # UDAL query input
    udal_query_input = pn.widgets.TextAreaInput(
        name='UDAL Query',
        placeholder='Enter UDAL query (e.g., season == "winter" AND depth > 100)',
        height=120,
        description="Use standard Python/Pandas query syntax"
    )
    
    # Quick filter presets
    preset_queries = {
        'Winter samples': 'season == "winter"',
        'Deep water (>100m)': 'depth > 100',
        'Surface samples': 'depth <= 10',
        'High salinity': 'sea_surface_salinity > 35',
        'Recent samples (2023+)': 'year >= 2023',
    }
    
    preset_selector = pn.widgets.Select(
        name='Quick Filters',
        options=['--'] + list(preset_queries.keys()),
        value='--',
        description="Select a query"
    )
    
    # Table selection for filtering
    table_selector = pn.widgets.CheckBoxGroup(
        name='Tables to filter',
        value=list(mgf_parquet_dfs.keys()) if mgf_parquet_dfs else [],
        options=list(mgf_parquet_dfs.keys()) if mgf_parquet_dfs else [],
        inline=True,
    )
    
    # Action buttons
    apply_metadata_button = pn.widgets.Button(
        name="Apply Filter on Metadata",
        button_type="primary",
        width=250,
    )
    
    clear_button = pn.widgets.Button(
        name="Clear Filter",
        button_type="warning",
        width=150,
    )
    
    validate_button = pn.widgets.Button(
        name="Validate Query",
        button_type="success",
        width=150,
    )

    filter_data_button = pn.widgets.Button(
        name="Filter data",
        button_type="primary",
        width=150,
    )
    
    # Status and results
    status_pane = pn.pane.Markdown(
        "**Status:** Ready to filter data",
        styles={'background': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'}
    )
    
    # Query history
    query_history_display = pn.widgets.Tabulator(
        pd.DataFrame(columns=['Query', 'Rows Before', 'Rows After', 'Status']),
        name='Query History',
        page_size=5,
        pagination='local',
        sizing_mode='stretch_width',
    )
    
    # Results indicators
    original_rows_indicator = pn.indicators.Number(
        name='Original Rows',
        value=len(full_metadata) if full_metadata is not None else 0,
        format='{value}',
        font_size='20pt',
        title_size='12pt',
        colors=[(0, 'gray')]
    )
    
    filtered_rows_indicator = pn.indicators.Number(
        name='Filtered Rows',
        value=0,
        format='{value}',
        font_size='20pt',
        title_size='12pt',
        colors=[(0, 'red'), (50, 'orange'), (100, 'green')]
    )
    
    reduction_indicator = pn.indicators.Number(
        name='Reduction %',
        value=0,
        format='{value:.1f}%',
        font_size='20pt',
        title_size='12pt',
        colors=[(0, 'green'), (50, 'orange'), (90, 'red')]
    )
    
    # Preview of filtered data
    preview_table = pn.widgets.Tabulator(
        pd.DataFrame(),
        name='Filtered Data Preview',
        page_size=10,
        pagination='local',
        sizing_mode='stretch_width',
    )
    
    # Available columns info
    if full_metadata is not None:
        columns_info = f"**Available columns ({len(full_metadata.columns)}):** {', '.join(sorted(full_metadata.columns[:10]))}..."
    else:
        columns_info = "**No data loaded yet**"
    
    columns_pane = pn.pane.Markdown(columns_info)
    
    # ========== CALLBACKS ==========
    
    def update_columns_info():
        """Update available columns information."""
        if state['original_metadata'] is not None:
            cols = sorted(state['original_metadata'].columns)
            if len(cols) > 20:
                cols_display = ', '.join(cols[:20]) + f'... ({len(cols)} total)'
            else:
                cols_display = ', '.join(cols)
            columns_pane.object = f"**Available columns ({len(cols)}):** {cols_display}"
    
    def preset_selected(event):
        """Update query input when preset is selected."""
        if event.new != 'Custom' and event.new in preset_queries:
            udal_query_input.value = preset_queries[event.new]
    
    preset_selector.param.watch(preset_selected, 'value')
    
    def validate_query(event=None):
        """Validate the UDAL query without applying it."""
        if state['original_metadata'] is None:
            status_pane.object = "**Status:** ⚠ No data loaded to validate against"
            return
        
        query = udal_query_input.value.strip()
        if not query:
            status_pane.object = "**Status:** ⚠ Please enter a query"
            return
        
        try:
            # Test the query on a small sample
            test_df = state['original_metadata'].head(100)
            result = test_df.query(query)
            
            status_pane.object = (
                f"**Status:** ✓ Query is valid! "
                f"Would return {len(result)}/100 rows in test sample"
            )
            logger.info(f"Query validated successfully: {query}")
            
        except Exception as e:
            status_pane.object = f"**Status:** ✗ Invalid query: {str(e)}"
            logger.error(f"Query validation failed: {e}")
    
    def apply_filter_metadata(event=None):
        """Apply the UDAL query filter to metadata only."""
        if state['original_metadata'] is None:
            status_pane.object = "**Status:** ⚠ No data loaded to filter"
            return
        
        query = udal_query_input.value.strip()
        if not query:
            status_pane.object = "**Status:** ⚠ Please enter a query"
            return
        
        try:
            original_count = len(state['original_metadata'])
            
            # Apply query to metadata
            status_pane.object = f"**Status:** ⏳ Applying filter to metadata..."
            filtered_meta = state['original_metadata'].query(query)
            filtered_count = len(filtered_meta)
            
            if filtered_count == 0:
                status_pane.object = "**Status:** ⚠ Query returned no results. Try a different filter."
                return
            
            state['filtered_metadata'] = filtered_meta
            
            # Update indicators
            filtered_rows_indicator.value = filtered_count
            reduction_pct = ((original_count - filtered_count) / original_count) * 100
            reduction_indicator.value = reduction_pct
            
            # Update preview
            preview_table.value = filtered_meta.head(50)
            
            # Update query history
            history_entry = {
                'Query': query,
                'Rows Before': original_count,
                'Rows After': filtered_count,
                'Status': '✓ Metadata filtered'
            }
            state['query_history'].append(history_entry)
            query_history_display.value = pd.DataFrame(state['query_history'])
            
            # Update status
            status_pane.object = (
                f"**Status:** ✓ Metadata filtered! "
                f"Reduced from {original_count} to {filtered_count} rows "
                f"({reduction_pct:.1f}% reduction). Click 'Filter data' to apply to data tables."
            )
            
            logger.info(
                f"Applied metadata filter '{query}': {original_count} → {filtered_count} rows"
            )
            
        except Exception as e:
            status_pane.object = f"**Status:** ✗ Filter failed: {str(e)}"
            logger.error(f"Metadata filter application failed: {e}", exc_info=True)
            
            # Add to history as failed
            history_entry = {
                'Query': query,
                'Rows Before': len(state['original_metadata']),
                'Rows After': 0,
                'Status': f'✗ {str(e)[:50]}'
            }
            state['query_history'].append(history_entry)
            query_history_display.value = pd.DataFrame(state['query_history'])
    
    def apply_filter_data(event=None):
        """Filter data tables based on already filtered metadata."""
        if state['original_metadata'] is None:
            status_pane.object = "**Status:** ⚠ No data loaded to filter"
            return
        
        if state['filtered_metadata'] is None:
            status_pane.object = "**Status:** ⚠ Please apply metadata filter first"
            return
        
        try:
            filtered_meta = state['filtered_metadata']
            
            # Filter mgf_parquet_dfs tables based on filtered metadata samples
            status_pane.object = f"**Status:** ⏳ Filtering data tables..."
            if state['original_tables'] is not None:
                state['filtered_tables'] = {}
                # Get sample IDs from filtered metadata
                sample_ids = set(filtered_meta.index)
                logger.info(sample_ids)
                
                # Log the filtering process
                logger.info(f"Filtering tables with {len(sample_ids)} sample IDs from metadata")
                
                tables_filtered = 0
                for table_name in table_selector.value:
                    if table_name in state['original_tables']:
                        df = state['original_tables'][table_name].copy()
                        original_rows = len(df)
                        
                        # Filter by index - handle both simple and MultiIndex
                        if isinstance(df.index, pd.MultiIndex):
                            # For MultiIndex, filter by first level (sample IDs)
                            filtered_df = df[df.index.get_level_values(0).isin(sample_ids)]
                        else:
                            # For simple index
                            filtered_df = df[df.index.isin(sample_ids)]
                        
                        state['filtered_tables'][table_name] = filtered_df
                        filtered_rows = len(filtered_df)
                        tables_filtered += 1
                        logger.info(f"Table '{table_name}': {original_rows} → {filtered_rows} rows")
                    else:
                        logger.warning(f"Table '{table_name}' not found in original tables")
            
            # Update status
            status_pane.object = (
                f"**Status:** ✓ Data tables filtered! "
                f"Filtered {tables_filtered} table(s) to match {len(sample_ids)} samples."
            )
            
            # Call callback if provided
            if on_filter_callback is not None:
                on_filter_callback(state)
            
            logger.info(
                f"Applied data filter: {tables_filtered} tables filtered to {len(sample_ids)} samples"
            )
            
        except Exception as e:
            status_pane.object = f"**Status:** ✗ Data filter failed: {str(e)}"
            logger.error(f"Data filter application failed: {e}", exc_info=True)
    
    def clear_filter(event=None):
        """Clear all filters and reset to original data."""
        state['filtered_metadata'] = None
        state['filtered_tables'] = None
        
        udal_query_input.value = ''
        preset_selector.value = 'Custom'
        
        original_count = len(state['original_metadata']) if state['original_metadata'] is not None else 0
        filtered_rows_indicator.value = original_count
        reduction_indicator.value = 0
        
        preview_table.value = pd.DataFrame()
        status_pane.object = "**Status:** Filters cleared. Showing original data."
        
        # Call callback to reset
        if on_filter_callback is not None:
            on_filter_callback(state)
        
        logger.info("Filters cleared")
    
    # Connect callbacks
    apply_metadata_button.on_click(apply_filter_metadata)
    filter_data_button.on_click(apply_filter_data)
    clear_button.on_click(clear_filter)
    validate_button.on_click(validate_query)
    
    # ========== LAYOUT ==========
    
    instructions = pn.pane.Markdown("""
    ### UDAL Query Filtering
    
    **Instructions:**
    1. Select a query
    2. Click **Validate Query** to test the syntax (optional)
    3. Select which tables to filter (LSU/SSU)
    4. Click **Apply Filter** to filter the data
    5. Use **Clear Filter** to reset to original data
    
    **UDAL Query Syntax:**
    - Use standard Python/Pandas query syntax
    - Examples:
      - `season == "winter"` - Exact match
      - `depth > 100` - Numerical comparison
      - `season.isin(["winter", "spring"])` - Multiple values
      - `(depth > 50) & (season == "winter")` - Combined conditions
      - `sample_id.str.contains("EMOBON")` - String matching
    
    **Tips:**
    - Column names with spaces need backticks: `` `observatory name` == "Oslo" ``
    - Use `&` for AND, `|` for OR, `~` for NOT
    - String comparisons are case-sensitive
    """)
    
    query_section = pn.Column(
        pn.pane.Markdown("#### Query Builder"),
        preset_selector,
        udal_query_input,
        pn.Row(validate_button, apply_metadata_button, clear_button),
        columns_pane,
    )
    
    table_selection = pn.Column(
        pn.pane.Markdown("#### Data Selection"),
        table_selector,
        filter_data_button,
    )
    
    indicators_row = pn.Row(
        original_rows_indicator,
        filtered_rows_indicator,
        reduction_indicator,
        sizing_mode='stretch_width',
    )
    
    udal_filter_tab = pn.Column(
        instructions,
        pn.layout.Divider(),
        query_section,
        pn.layout.Divider(),
        status_pane,
        indicators_row,
        pn.layout.Divider(),
        pn.pane.Markdown("#### Query History"),
        query_history_display,
        pn.layout.Divider(),
        table_selection,
        # pn.layout.Divider(),
        # pn.pane.Markdown("#### Filtered Data Preview"),
        # preview_table,
        scroll=True,
        sizing_mode='stretch_width',
    )
    
    # Initialize
    update_columns_info()
    
    return udal_filter_tab, state
