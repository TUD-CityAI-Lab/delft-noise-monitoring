import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analysis_paths import connect_analysis, data_root

def _connect_duckdb():
    """Connect read-only to DELFT_DATABASE, or the repository's analysis.duckdb."""
    return connect_analysis()


def source_cooccurrence(db, source_cols, threshold=0.5):
    """P(j | i) from strict-threshold joint counts, without loading the lake.

    NULL scores are not detections. A class with no detections has an undefined
    (NaN) row. Frames with no detections contribute to neither the conditional
    numerator nor denominator.
    """
    available = {row[0] for row in db.execute('DESCRIBE sources').fetchall()}
    if not source_cols or len(set(source_cols)) != len(source_cols) or not set(source_cols) <= available:
        raise ValueError('Source columns must be nonempty, unique, and present in sources')
    if not np.isfinite(threshold):
        raise ValueError('Threshold must be finite')
    predicates = [f'"{name.replace(chr(34), chr(34)*2)}" > {float(threshold)}' for name in source_cols]
    expressions = [f'COUNT(*) FILTER (WHERE {condition})' for condition in predicates]
    expressions += [f'COUNT(*) FILTER (WHERE ({left}) AND ({right}))'
                    for left in predicates for right in predicates]
    counts = np.asarray(db.execute('SELECT '+', '.join(expressions)+' FROM sources').fetchone(), dtype=np.int64)
    size = len(source_cols)
    matrix = np.full((size, size), np.nan)
    np.divide(counts[size:].reshape(size, size), counts[:size, None],
              out=matrix, where=counts[:size, None] != 0)
    return pd.DataFrame(matrix, index=source_cols, columns=source_cols)

def plot_sensors_on_map_df(
    df,
    sensor_id_col: str = 'sensor_id',
    dot_size: int = 15,
    width: int = 900,
    height: int = 600,
    title: str = "Sensor Locations (Interactive Map)",
    color_col: str | None = None,
    color_scale: str | list = "Viridis",
    range_color: tuple[float, float] | None = None,
    add_id_label:  bool = False,
    extra_pois = None,
    legend_tickfont_size: int = 16,
    outline: bool = False,
    binnenstad_polygon = False
):
    """
    Plot sensors on a map using a dataframe that contains sensor data.

    Parameters:
    - df: DataFrame with sensor data (must include sensor_id_col column)
    - sensor_id_col: Name of the column containing sensor IDs
    - dot_size: Size of the dots on the map
    - width: Width of the map in pixels
    - title: Title for the map
    - color_col: Optional. Name of a numeric column in df to control color intensity of dots
                 (continuous colorbar). If provided and numeric, a continuous scale is used.
    - color_scale: Plotly color scale name or list; used when color_col is provided (default: "Viridis")
    - range_color: Optional (min, max) tuple to fix the color domain when color_col is provided
    - add_id_label: If True, adds a label with the last 2 characters of the sensor ID next to each dot.
    - legend_tickfont_size: Font size for the legend.
    - outline: If True, adds an outline to each dot.
    - binnenstad_polygon: If True, adds a polygon outline of the historic center to the map.
    The function will merge with sensor metadata and display all columns from df in hover data.
    """
    # Load sensor metadata from DuckDB `sensors` table
    con = _connect_duckdb()
    sensors_meta = (
        con.sql("SELECT sensor_id, latitude, longitude FROM sensors").df()
    )

    plot_df = df.merge(
        sensors_meta[["sensor_id", "latitude", "longitude"]],
        left_on=sensor_id_col,
        right_on="sensor_id",
        how="inner",
    )

    if len(plot_df) == 0:
        print("Warning: No sensors matched between dataframe and metadata")
        return

    # Extract last 2 digits for label (handles both 'citizen-001' and '001')
    def last2(sid):
        return str(sid)[-2:]

    if add_id_label:
        plot_df['label'] = plot_df['sensor_id'].apply(last2)

    use_continuous_color = False
    if color_col is not None:
        if color_col not in plot_df.columns:
            raise KeyError(f"color_col '{color_col}' not found in provided dataframe")
        if pd.api.types.is_numeric_dtype(plot_df[color_col]):
            use_continuous_color = True
        else:
            coerced = pd.to_numeric(plot_df[color_col], errors='coerce')
            if coerced.notna().any():
                plot_df[color_col] = coerced
                use_continuous_color = True
            else:
                print(f"Note: color_col '{color_col}' is non-numeric; using discrete colors instead of intensity.")

    # Calculate center as mean of extreme locations (bounding box center)
    center_lat = (plot_df['latitude'].min() + plot_df['latitude'].max()) / 2
    center_lon = (plot_df['longitude'].min() + plot_df['longitude'].max()) / 2


    hover_cols = {}
    for col in plot_df.columns:
        if col not in ['latitude', 'longitude', sensor_id_col, 'sensor_id']:
            if pd.api.types.is_numeric_dtype(plot_df[col]):
                hover_cols[col] = ':.2f'
            else:
                hover_cols[col] = True
    hover_cols['latitude'] = ':.5f'
    hover_cols['longitude'] = ':.5f'

    scatter_kwargs = dict(
        data_frame=plot_df,
        lat="latitude",
        lon="longitude",
        hover_name=sensor_id_col,
        hover_data=hover_cols,
        size_max=dot_size,
        height=600,
    )

    # Make text labels optional: only include if 'label' column is present
    if 'label' in plot_df.columns:
        scatter_kwargs["text"] = "label"

    if color_col is not None:
        scatter_kwargs["color"] = color_col
        if use_continuous_color:
            scatter_kwargs["color_continuous_scale"] = color_scale
            if range_color is not None:
                scatter_kwargs["range_color"] = range_color
        else:
            pass
    else:
        scatter_kwargs["color_discrete_sequence"] = ["red"]

    fig = px.scatter_map(**scatter_kwargs)
    fig.update_traces(marker=dict(size=dot_size, opacity=1), textposition="top center", textfont=dict(color="black", size=12))

    # Simulate outline by adding larger black dots behind the original ones
    if outline:
        outline_trace = go.Scattermap(
            lat=plot_df["latitude"],
            lon=plot_df["longitude"],
            mode="markers",
            marker=dict(size=dot_size + 2, color="black", opacity=1),
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_trace(outline_trace)
        # Move outline trace to the back (first in data list)
        fig.data = (fig.data[-1],) + fig.data[:-1]

    if binnenstad_polygon:
        binnenstad_outline_path = data_root() / "ref" / "delft-binnenstad-outline.csv"
        binnenstad_df = pd.read_csv(binnenstad_outline_path)
        polygon_trace = go.Scattermap(
            lat=binnenstad_df["latitude"],
            lon=binnenstad_df["longitude"],
            mode="lines",
            line=dict(color="#ff5050", width=1.5),
            fill="toself",
            fillcolor="rgba(100, 100, 100, 0.12)",
            name="Binnenstad",
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_trace(polygon_trace)
        fig.data = (fig.data[-1],) + fig.data[:-1]

    # Hide the column name label on the color scale / legend (keep the scale itself).
    if color_col is not None:
        if use_continuous_color:
            fig.update_coloraxes(
                colorbar_title_text="",
                colorbar_tickfont_size=legend_tickfont_size,
            )
        else:
            fig.update_layout(
                legend_title_text="",
                legend=dict(font=dict(size=legend_tickfont_size)),
            )
            
    if extra_pois is not None:
        poi_scatter = px.scatter_map(
            extra_pois,
            lat="lat",
            lon="lon",
            hover_data=extra_pois,
            color_discrete_sequence=["purple"],
            size_max=dot_size,
        )
        for trace in poi_scatter.data:
            trace.marker.size = dot_size + 3
            trace.marker.symbol = "star"
            trace.marker.color = "orange"
            trace.name = "POI"
            fig.add_trace(trace)
        
    layout_kwargs = {
        "map": {
            "style": "open-street-map",
            "zoom": 12.0,
            "center": {"lat": center_lat, "lon": center_lon},
        },
        "margin": {"r": 0, "t": 40, "l": 0, "b": 0},
        "title": title,
    }
    if width:
        layout_kwargs["width"] = width
    if height:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.show()


def plot_source_avg_per_minute(
    source: str,
    threshold: float = 0.5,
    *,
    color: str = "royalblue",
    window_size: int | None = None,
    window_color: str = "black",
    window_center: bool = True,
    title: str | None = None,
    width: int = 1000,
    height: int = 450,
    base_date: str = "2000-01-01",
    show: bool = True,
    return_df: bool = False,
    plot_median_dba: bool = False,
    rangeslider_visible: bool = True,
    diurnal: bool = True,
    font_size: int = 14,
):
    """Plot pooled sound-event presence percentages by minute of day.

    Presence is 100 * total detections / total recorded fragments at that minute,
    pooled across sensors and dates (fragment weighting, not equal date weighting).
    Detection uses score > threshold. All timestamped fragments enter the denominator,
    including NULL scores, which are not positive detections. Observed zero-detection
    minutes remain zero; absent minutes are not fabricated. The rolling overlay is
    an arithmetic mean of minute percentages. Detection-conditioned loudness and
    legacy count columns are retained for comparison with historical outputs.

    Parameters
    - source: Name of the source probability column (e.g., "church", "siren").
    - threshold: Probability threshold to consider a frame as a detection.
    - color: Line color for the main plot.
    - window_size: Optional size (in minutes/points) for a centered rolling average overlay. If None, no overlay.
    - window_color: Color for the rolling average line.
    - window_center: Whether to use centered=True for the rolling window.
    - title: Optional title.
    - width, height: Figure dimensions.
    - base_date: Reference date for plotting time-of-day on a time axis.
    - show: If True, displays the figure. Set False to return the figure without displaying.
    - return_df: If True, also returns the underlying aggregated DataFrame.
    - plot_median_dba: If True, plot also contains the median dba
    - rangeslider_visible: Whether to show the range slider below the x-axis.
    - diurnal: If True, formats the x-axis for a full 24-hour day with ticks at regular intervals.
    - font_size: Font size for axis tick labels and axis titles (default: 14).

    Returns
    - fig or (fig, df): Plotly Figure; if return_df=True, returns a tuple (fig, DataFrame).
    """
    con = _connect_duckdb()

    # Validate that the requested column exists by peeking at schema (no data scan)
    try:
        schema_cols = con.sql('SELECT * FROM sources LIMIT 0').df().columns.tolist()
    except Exception as e:
        raise RuntimeError(
            'Failed to read the sources view in the selected database. Check prepared-data setup.'
        ) from e

    if source not in schema_cols:
        raise ValueError(
            f"Column '{source}' not found in sources dataset. Available columns include: "
            + ", ".join(schema_cols[:25]) + (" ..." if len(schema_cols) > 25 else "")
        )

    # Pool actual fragment denominators; retain observed zero-detection groups.
    sql = f"""
        WITH frames AS (
            SELECT
            date,
            time,
            avg_spl_a,
            "{source}" > ? AS detected,
            CAST(strftime(time, '%H') AS INTEGER) * 60
                + CAST(strftime(time, '%M') AS INTEGER) AS minute_of_day
            FROM sources
            WHERE time IS NOT NULL
        ),
        per_day AS (
            SELECT date, minute_of_day,
                COUNT(*) FILTER (WHERE detected) AS hits,
                COUNT(*) AS n_fragments,
                COUNT(*) FILTER (WHERE detected IS NULL) AS n_null_scores,
                MEDIAN(avg_spl_a) FILTER (WHERE detected) AS spla
            FROM frames
            GROUP BY date, minute_of_day
        ),
        agg AS (
            SELECT minute_of_day,
                SUM(hits)::BIGINT AS detections,
                SUM(n_fragments)::BIGINT AS n_fragments,
                SUM(n_null_scores)::BIGINT AS n_null_scores,
                100.0 * SUM(hits) / SUM(n_fragments) AS presence_pct,
                AVG(hits) AS avg_hits_per_day,
                10 * LOG10(MEDIAN(spla)) AS median_dba
            FROM per_day
            GROUP BY minute_of_day
        )
        SELECT
            minute_of_day,
            LPAD(CAST(CAST(FLOOR(minute_of_day / 60) AS INTEGER) AS VARCHAR), 2, '0') || ':' ||
            LPAD(CAST(minute_of_day % 60 AS VARCHAR), 2, '0') AS hhmm,
            detections, n_fragments, n_null_scores, presence_pct,
            avg_hits_per_day,
            median_dba
        FROM agg
        ORDER BY minute_of_day
    """

    df = con.execute(sql, [float(threshold)]).df()

    # Build time-of-day axis on a fixed base date for interactivity
    base = pd.Timestamp(base_date)
    df["tod"] = base + pd.to_timedelta(df["minute_of_day"], unit="m")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tod"],
            y=df["presence_pct"],
            mode="lines",
            name=f"{source} (presence %)",
            line=dict(color=color),
        )
    )

    if plot_median_dba:
        fig.add_trace(
            go.Scatter(
                x=df["tod"],
                y=df["median_dba"],
                mode="lines",
                name="median dBA",
                line=dict(color="firebrick"),
                yaxis="y2",
            )
        )

    # Optional rolling window overlay (computed on sorted minute_of_day order)
    if window_size is not None and int(window_size) > 1:
        df_sorted = df.sort_values("minute_of_day").copy()
        # Use a forgiving min_periods to avoid dropping edges entirely
        minp = max(1, int(window_size) // 3)
        df_sorted["avg_hits_smooth"] = (
            df_sorted["avg_hits_per_day"].rolling(
                window=int(window_size), center=bool(window_center), min_periods=minp
            ).mean()
        )
        df_sorted["presence_pct_smooth"] = df_sorted["presence_pct"].rolling(
            window=int(window_size), center=bool(window_center), min_periods=minp
        ).mean()
        # Merge back to original order for plotting aligned with tod
        df = df.merge(
            df_sorted[["minute_of_day", "avg_hits_smooth", "presence_pct_smooth"]],
            on="minute_of_day",
            how="left",
        )
        fig.add_trace(
            go.Scatter(
                x=df["tod"],
                y=df["presence_pct_smooth"],
                mode="lines",
                name=f"{source} (rolling {int(window_size)}m)",
                line=dict(color=window_color, width=3, dash="dot"),
            )
        )

    if rangeslider_visible:
        rangeselector_dict = dict(
            buttons=[
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all"),
            ]
        )
    else:
        rangeselector_dict = None

    fig.update_layout(
        title_text=title,
        xaxis=dict(
            title=dict(text="Time of day", font=dict(size=font_size)),
            rangeselector=rangeselector_dict,
            rangeslider=dict(visible=rangeslider_visible),
            type="date",
            tickformat="%H:%M",
            tickfont=dict(size=font_size),
        ),
        yaxis=dict(
            title=dict(text="Presence ratio (%)", font=dict(size=font_size)),
            showgrid=True,
            tickfont=dict(size=font_size),
        ),
        yaxis2=(
            dict(
                title=dict(text="Median dBA", font=dict(size=font_size)),
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(size=font_size),
            )
            if plot_median_dba
            else None
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        width=width,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    
    # ensure legend text honors font_size
    fig.update_layout(legend=dict(font=dict(size=font_size)))

    if diurnal:
        # x-axis should cover a full day including ticks at "00:00" "24:00" for improved readability.
        tickvals = [pd.Timedelta(hours=h) for h in range(0, 24+1, int(24/6))]
        ticktext = [f"{h:02d}:00" for h in range(0, 24, int(24/6))] + ["24:00"]
        fig.update_xaxes(
            tickvals=[base + tv for tv in tickvals],
            ticktext=ticktext,
        )
        fig.update_xaxes(range=[base, base + pd.Timedelta(days=1)])

    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            # 'height': 500,
            # 'width': 700,
            'scale': 4
        }
    }
    
    fig.update_layout(margin=dict(l=5, r=5, t=4, b=5))
    
    if show:
        fig.show(config=config)

    return (fig, df) if return_df else fig
