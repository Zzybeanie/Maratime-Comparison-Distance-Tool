# %pip install searoute folium pandas pyproj streamlit streamlit_folium plotly geopy
import pandas as pd
import numpy as np
import folium
import streamlit as st
import plotly.graph_objects as go
from math import radians, sin, cos, asin, sqrt
from difflib import get_close_matches
from pyproj import Geod
import searoute as sr
from streamlit_folium import st_folium
from typing import Tuple, Optional, List
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

# ---------- Page config ----------
st.set_page_config(
    page_title="Maritime Route Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .port-match-exact {
        color: #28a745;
        font-weight: 600;
    }
    .port-match-fuzzy {
        color: #ffc107;
        font-weight: 600;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Data loading ----------
@st.cache_data
def load_ports_data():
    """Load and process ports data with proper error handling."""
    try:
        ports = pd.read_csv("ports_with_latlong.csv")
        
        def split_decimal_latlon(s):
            if pd.isna(s):
                return (np.nan, np.nan)
            try:
                cleaned = str(s).strip().replace(" ", "")
                if "," in cleaned:
                    lat_s, lon_s = cleaned.split(",", 1)
                else:
                    parts = cleaned.split()
                    if len(parts) >= 2:
                        lat_s, lon_s = parts[0], parts[1]
                    else:
                        return (np.nan, np.nan)
                
                lat, lon = float(lat_s.strip()), float(lon_s.strip())
                
                # Validate coordinate ranges
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    return (np.nan, np.nan)
                
                return lat, lon
            except Exception:
                return (np.nan, np.nan)
        
        latlon = ports["lat_long"].apply(split_decimal_latlon)
        ports["lat"] = [t[0] for t in latlon]
        ports["lon"] = [t[1] for t in latlon]
        ports["port_uc"] = ports["port_name"].str.strip().str.upper()
        ports = ports.drop_duplicates(subset=["port_uc"], keep="first")
        
        valid_ports = ports[(ports["lat"].notna()) & (ports["lon"].notna())]
        
        if valid_ports.empty:
            st.error("❌ No valid ports found in the CSV file.")
        
        return valid_ports
    except FileNotFoundError:
        st.error("❌ **ports_with_latlong.csv** file not found. Please place it in the same directory as app.py.")
        return pd.DataFrame(columns=["port_name", "lat_long", "lat", "lon", "port_uc"])
    except Exception as e:
        st.error(f"❌ Error loading ports data: {str(e)}")
        return pd.DataFrame(columns=["port_name", "lat_long", "lat", "lon", "port_uc"])

# ---------- Session state ----------
if 'ports_df' not in st.session_state:
    st.session_state.ports_df = load_ports_data()
if 'origin_port' not in st.session_state:
    st.session_state.origin_port = "AMBON"
if 'dest_port' not in st.session_state:
    st.session_state.dest_port = "BIAK"
if 'speed_kts' not in st.session_state:
    st.session_state.speed_kts = 12.0
if 'map_key' not in st.session_state:
    st.session_state.map_key = 0
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False
if 'origin_match_type' not in st.session_state:
    st.session_state.origin_match_type = None
if 'dest_match_type' not in st.session_state:
    st.session_state.dest_match_type = None
if 'geocoder' not in st.session_state:
    st.session_state.geocoder = Nominatim(user_agent="maritime_route_app")
if 'country_cache' not in st.session_state:
    st.session_state.country_cache = {}
if 'use_manual_origin' not in st.session_state:
    st.session_state.use_manual_origin = False
if 'use_manual_dest' not in st.session_state:
    st.session_state.use_manual_dest = False
if 'manual_origin_name' not in st.session_state:
    st.session_state.manual_origin_name = ""
if 'manual_origin_lat' not in st.session_state:
    st.session_state.manual_origin_lat = 0.0
if 'manual_origin_lon' not in st.session_state:
    st.session_state.manual_origin_lon = 0.0
if 'manual_dest_name' not in st.session_state:
    st.session_state.manual_dest_name = ""
if 'manual_dest_lat' not in st.session_state:
    st.session_state.manual_dest_lat = 0.0
if 'manual_dest_lon' not in st.session_state:
    st.session_state.manual_dest_lon = 0.0

# ---------- Helpers ----------
def get_country_from_coords(lat: float, lon: float) -> str:
    """Get country name from coordinates using reverse geocoding."""
    cache_key = f"{lat:.2f},{lon:.2f}"
    
    # Check cache first
    if cache_key in st.session_state.country_cache:
        return st.session_state.country_cache[cache_key]
    
    try:
        geolocator = Nominatim(user_agent="maritime_route_comparison_tool_v1", timeout=10)
        location = geolocator.reverse(f"{lat}, {lon}", language='en', exactly_one=True)
        
        if location and location.raw.get('address'):
            country = location.raw['address'].get('country', 'Unknown')
            if country and country != 'Unknown':
                st.session_state.country_cache[cache_key] = country
                return country
        
        # If no country found, try without language specification
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
        if location and location.raw.get('address'):
            country = location.raw['address'].get('country', 'Unknown')
            if country and country != 'Unknown':
                st.session_state.country_cache[cache_key] = country
                return country
        
        return "Unknown"
    except GeocoderTimedOut:
        # Retry once on timeout
        try:
            time.sleep(1)
            geolocator = Nominatim(user_agent="maritime_route_comparison_tool_v1", timeout=10)
            location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
            if location and location.raw.get('address'):
                country = location.raw['address'].get('country', 'Unknown')
                st.session_state.country_cache[cache_key] = country
                return country
        except Exception:
            return "Unknown"
    except GeocoderServiceError:
        return "Unknown"
    except Exception as e:
        # For debugging - you can remove this later
        print(f"Geocoding error for {lat}, {lon}: {str(e)}")
        return "Unknown"

def get_coords(port_name: str) -> Tuple[Optional[float], Optional[float], Optional[str], str, str]:
    """Return (lat, lon, pretty_name, match_type, country) for a port name with fuzzy matching."""
    if st.session_state.ports_df.empty:
        return None, None, None, "error", "Unknown"
    
    q = str(port_name).strip().upper()
    
    # Exact match
    hit = st.session_state.ports_df.loc[st.session_state.ports_df["port_uc"] == q]
    if not hit.empty:
        r = hit.iloc[0]
        lat, lon = float(r["lat"]), float(r["lon"])
        country = get_country_from_coords(lat, lon)
        return lat, lon, r["port_name"], "exact", country
    
    # Fuzzy match
    choices = st.session_state.ports_df["port_uc"].tolist()
    cand = get_close_matches(q, choices, n=1, cutoff=0.6)
    if cand:
        r = st.session_state.ports_df.loc[st.session_state.ports_df["port_uc"] == cand[0]].iloc[0]
        lat, lon = float(r["lat"]), float(r["lon"])
        country = get_country_from_coords(lat, lon)
        return lat, lon, r["port_name"], "fuzzy", country
    
    # Partial match
    partial_matches = st.session_state.ports_df[st.session_state.ports_df["port_uc"].str.contains(q, na=False)]
    if not partial_matches.empty:
        r = partial_matches.iloc[0]
        lat, lon = float(r["lat"]), float(r["lon"])
        country = get_country_from_coords(lat, lon)
        return lat, lon, r["port_name"], "partial", country
    
    return None, None, None, "error", "Unknown"

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance using Haversine formula (NM)."""
    R_km = 6371.0088
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = (φ2 - φ1), (λ2 - λ1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    km = R_km * 2 * asin(sqrt(a))
    return km / 1.852

# --- Improved geodesic handling ---
GEOD = Geod(ellps="WGS84")

def normalize_longitude(lon: float) -> float:
    """Normalize longitude to [-180, 180] range."""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon

def geodesic_polyline_smart(lat1: float, lon1: float, lat2: float, lon2: float, n: int = 100) -> List[List[float]]:
    """
    Generate geodesic polyline with intelligent anti-meridian handling.
    Returns segments that properly handle dateline crossing.
    """
    # Normalize longitudes
    lon1 = normalize_longitude(lon1)
    lon2 = normalize_longitude(lon2)
    
    # Calculate the geodesic
    geod_line = GEOD.inv_intermediate(lon1, lat1, lon2, lat2, n)
    
    segments = []
    current_segment = []
    
    for i, (lon, lat) in enumerate(zip(geod_line.lons, geod_line.lats)):
        lon_norm = normalize_longitude(lon)
        
        if i > 0:
            prev_lon = normalize_longitude(geod_line.lons[i-1])
            lon_diff = abs(lon_norm - prev_lon)
            
            # Detect anti-meridian crossing (jump > 180°)
            if lon_diff > 180:
                # Finish current segment
                if current_segment:
                    segments.append(current_segment)
                # Start new segment
                current_segment = [[lat, lon_norm]]
            else:
                current_segment.append([lat, lon_norm])
        else:
            current_segment.append([lat, lon_norm])
    
    # Add the last segment
    if current_segment:
        segments.append(current_segment)
    
    return segments

# ---------- Core map builder ----------
def create_route_map(origin_name: str, dest_name: str, speed_kts: float = 12.0):
    """Create interactive map with route comparison."""
    try:
        # Get origin coordinates
        if st.session_state.use_manual_origin:
            lat_o = st.session_state.manual_origin_lat
            lon_o = st.session_state.manual_origin_lon
            origin_print = st.session_state.manual_origin_name
            origin_match = "manual"
            origin_country = get_country_from_coords(lat_o, lon_o)
        else:
            lat_o, lon_o, origin_print, origin_match, origin_country = get_coords(origin_name)
        
        # Get destination coordinates
        if st.session_state.use_manual_dest:
            lat_d = st.session_state.manual_dest_lat
            lon_d = st.session_state.manual_dest_lon
            dest_print = st.session_state.manual_dest_name
            dest_match = "manual"
            dest_country = get_country_from_coords(lat_d, lon_d)
        else:
            lat_d, lon_d, dest_print, dest_match, dest_country = get_coords(dest_name)
        
        # Store match types and countries for validation display
        st.session_state.origin_match_type = origin_match
        st.session_state.dest_match_type = dest_match
        st.session_state.origin_resolved = origin_print
        st.session_state.dest_resolved = dest_print
        st.session_state.origin_country = origin_country
        st.session_state.dest_country = dest_country
        
        if None in [lat_o, lon_o, lat_d, lon_d]:
            error_parts = []
            if origin_match == "error":
                error_parts.append(f"**Origin port '{origin_name}' not found**")
            if dest_match == "error":
                error_parts.append(f"**Destination port '{dest_name}' not found**")
            return None, " and ".join(error_parts) + ".\n\nPlease check port names in the dropdown.", None, None, None, None, ""
        
        # Calculate Haversine (great circle) distance
        haversine_dist_nm = haversine_nm(lat_o, lon_o, lat_d, lon_d)
        haversine_duration_hr = haversine_dist_nm / speed_kts if speed_kts > 0 else 0
        
        # Check if route is too short
        if haversine_dist_nm < 10:
            return None, f"⚠️ **Route too short** ({haversine_dist_nm:.1f} NM).\n\nPlease select ports at least 10 NM apart for meaningful comparison.", None, None, None, None, ""
        
        # SeaRoute attempt
        sr_dist_nm = 0.0
        sr_duration_hr = 0.0
        sr_coords_latlon: List[List[float]] = []
        sr_success = False
        sr_error_msg = ""
        
        try:
            speed_knot_int = max(1, int(round(speed_kts)))
            feat = sr.searoute([lon_o, lat_o], [lon_d, lat_d],
                             speed_knot=speed_knot_int, units="naut")
            props = feat["properties"]
            path = feat["geometry"]["coordinates"]
            sr_dist_nm = float(props.get("length", 0.0))
            sr_duration_hr = float(props.get("duration_hours", 0.0))
            sr_coords_latlon = [[pt[1], pt[0]] for pt in path]
            sr_success = True
        except Exception as e:
            sr_success = False
            error_str = str(e).lower()
            
            # Better error messages
            if "no route found" in error_str or "route" in error_str:
                sr_error_msg = "⚠️ **SeaRoute calculation failed:** No navigable maritime route found\n\n**Possible reasons:**\n- One or both ports may be landlocked or in isolated water bodies\n- Route crosses closed/restricted waters\n- Ports are not connected by navigable waters\n\n✅ **Haversine (great circle) route is still shown** for reference as the theoretical shortest path."
            elif "timeout" in error_str:
                sr_error_msg = "⚠️ **SeaRoute calculation failed:** Request timeout\n\n**Possible reasons:**\n- Route is very complex to calculate\n- SeaRoute server is experiencing high load\n\n💡 **Suggestion:** Try again in a few moments.\n\n✅ **Haversine route is still shown** for reference."
            elif "connection" in error_str or "network" in error_str:
                sr_error_msg = "⚠️ **SeaRoute calculation failed:** Network connection issue\n\n**Please check:**\n- Your internet connection\n- SeaRoute API availability\n\n✅ **Haversine route is still shown** for reference."
            else:
                sr_error_msg = f"⚠️ **SeaRoute calculation failed:** {str(e)}\n\n✅ **Haversine route is still shown** for reference."
        
        # Calculate map center and zoom
        center_lat = (lat_o + lat_d) / 2
        center_lon = (lon_o + lon_d) / 2
        
        # Adaptive zoom based on distance
        if haversine_dist_nm < 100:
            zoom = 8
        elif haversine_dist_nm < 500:
            zoom = 6
        elif haversine_dist_nm < 1000:
            zoom = 5
        elif haversine_dist_nm < 2000:
            zoom = 4
        else:
            zoom = 3
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            tiles="CartoDB positron",
            zoom_start=zoom,
            control_scale=True,
            prefer_canvas=True
        )
        
        # Add markers
        folium.Marker(
            [lat_o, lon_o],
            tooltip=f"Origin: {origin_print}, {origin_country}",
            popup=f"<b>Origin:</b> {origin_print}<br><b>Country:</b> {origin_country}<br><b>Lat:</b> {lat_o:.4f}<br><b>Lon:</b> {lon_o:.4f}",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)
        
        folium.Marker(
            [lat_d, lon_d],
            tooltip=f"Destination: {dest_print}, {dest_country}",
            popup=f"<b>Destination:</b> {dest_print}<br><b>Country:</b> {dest_country}<br><b>Lat:</b> {lat_d:.4f}<br><b>Lon:</b> {lon_d:.4f}",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
        ).add_to(m)
        
        # Draw Haversine (geodesic) route - IMPROVED ANTI-MERIDIAN HANDLING
        geodesic_segments = geodesic_polyline_smart(lat_o, lon_o, lat_d, lon_d, n=150)
        
        for segment in geodesic_segments:
            if len(segment) > 1:  # Only draw if segment has multiple points
                folium.PolyLine(
                    segment,
                    color="#FF6B6B",
                    weight=3,
                    opacity=0.8,
                    tooltip=f"Haversine (Great Circle): {haversine_dist_nm:,.1f} NM"
                ).add_to(m)
        
        # Draw SeaRoute if available
        if sr_success and sr_coords_latlon:
            folium.PolyLine(
                sr_coords_latlon,
                color="#51CF66",
                weight=4,
                opacity=0.9,
                tooltip=f"SeaRoute (Navigable): {sr_dist_nm:,.1f} NM"
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <style>
        #route-legend {
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 9999;
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
            color: #000000;
        }
        #route-legend .title,
        #route-legend .item {
            color: #000000 !important;
        }
        #route-legend .title {
            font-weight: 700;
            margin-bottom: 10px;
            font-size: 14px;
        }
        #route-legend .item {
            margin: 5px 0;
            font-size: 13px;
        }
        </style>
        <div id="route-legend">
        <div class="title">📍 Route Types</div>
        <div class="item">
            <span style="color:#51CF66;">━━━</span> SeaRoute (Navigable)
        </div>
        <div class="item">
            <span style="color:#FF6B6B;">━━━</span> Haversine (Great Circle)
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Smart bounds fitting
        try:
            all_points = [[lat_o, lon_o], [lat_d, lon_d]]
            if sr_success and sr_coords_latlon:
                all_points.extend(sr_coords_latlon)
            
            lats = [p[0] for p in all_points]
            lons = [p[1] for p in all_points]
            
            sw = [min(lats) - 2, min(lons) - 2]
            ne = [max(lats) + 2, max(lons) + 2]
            
            m.fit_bounds([sw, ne])
        except Exception:
            pass
        
        return m, sr_dist_nm if sr_success else None, sr_duration_hr if sr_success else None, \
               haversine_dist_nm, haversine_duration_hr, sr_success, sr_error_msg
        
    except Exception as e:
        return None, f"❌ **Unexpected error:** {str(e)}\n\nPlease check your inputs and try again.", None, None, None, None, ""

def create_comparison_chart(sr_dist, haversine_dist, sr_dur, haversine_dur):
    """Create interactive comparison charts using Plotly."""
    
    # Distance comparison
    fig_dist = go.Figure(data=[
        go.Bar(
            x=['SeaRoute', 'Haversine'],
            y=[sr_dist or 0, haversine_dist],
            marker_color=['#51CF66', '#FF6B6B'],
            text=[f'{sr_dist:,.1f} NM' if sr_dist else 'N/A', f'{haversine_dist:,.1f} NM'],
            textposition='auto',
        )
    ])
    
    fig_dist.update_layout(
        title='Distance Comparison',
        yaxis_title='Distance (Nautical Miles)',
        showlegend=False,
        height=300,
        template='plotly_white'
    )
    
    # Duration comparison
    fig_dur = go.Figure(data=[
        go.Bar(
            x=['SeaRoute', 'Haversine'],
            y=[sr_dur or 0, haversine_dur],
            marker_color=['#51CF66', '#FF6B6B'],
            text=[f'{sr_dur:,.1f}h' if sr_dur else 'N/A', f'{haversine_dur:,.1f}h'],
            textposition='auto',
        )
    ])
    
    fig_dur.update_layout(
        title='Duration Comparison',
        yaxis_title='Duration (Hours)',
        showlegend=False,
        height=300,
        template='plotly_white'
    )
    
    return fig_dist, fig_dur

# ---------- Actions ----------
def calculate_route():
    """Calculate route with comprehensive error handling."""
    try:
        with st.spinner("🧭 Calculating optimal route..."):
            result = create_route_map(
                st.session_state.origin_port,
                st.session_state.dest_port,
                st.session_state.speed_kts
            )
        
        if result[0] is not None:
            (st.session_state.map_obj,
             st.session_state.sr_dist,
             st.session_state.sr_dur,
             st.session_state.haversine_dist,
             st.session_state.haversine_dur,
             st.session_state.sr_success,
             st.session_state.sr_error) = result
            st.session_state.calculation_done = True
            st.session_state.map_key += 1
            st.success("✅ Route calculation completed!")
        else:
            error_msg = result[1] if len(result) > 1 else "Unknown error"
            st.error(error_msg)
            st.session_state.calculation_done = False
    except Exception as e:
        st.error(f"❌ **An unexpected error occurred:** {str(e)}\n\nPlease check your inputs and try again.")
        st.session_state.calculation_done = False

def swap_ports():
    """Swap origin and destination ports."""
    st.session_state.origin_port, st.session_state.dest_port = \
        st.session_state.dest_port, st.session_state.origin_port

# ---------- UI ----------
st.markdown('<p class="main-header">🌊 Maritime Route Comparison Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Compare SeaRoute (navigable) vs Haversine (great circle) distances</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Route Parameters")
    
    # Origin Port Selection
    st.subheader("🟢 Origin Port")
    use_manual_origin = st.checkbox("Enter manually", key="manual_origin_check", value=st.session_state.use_manual_origin)
    st.session_state.use_manual_origin = use_manual_origin
    
    if use_manual_origin:
        st.session_state.manual_origin_name = st.text_input("Port Name", value=st.session_state.manual_origin_name, key="manual_origin_name_input", placeholder="e.g., Custom Port")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.session_state.manual_origin_lat = st.number_input("Latitude", value=st.session_state.manual_origin_lat, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f", key="manual_origin_lat_input")
        with col_o2:
            st.session_state.manual_origin_lon = st.number_input("Longitude", value=st.session_state.manual_origin_lon, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f", key="manual_origin_lon_input")
    else:
        if not st.session_state.ports_df.empty:
            port_names = sorted(st.session_state.ports_df["port_name"].dropna().unique())
            origin_port = st.selectbox(
                "Select from list",
                port_names,
                index=port_names.index(st.session_state.origin_port) if st.session_state.origin_port in port_names else 0,
                key="origin_select"
            )
            st.session_state.origin_port = origin_port
        else:
            st.warning("⚠️ No port data available")
            st.session_state.origin_port = st.text_input("Origin Port", st.session_state.origin_port)
    
    # Swap button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄", help="Swap origin and destination ports", use_container_width=True):
            # Swap CSV ports or manual inputs
            if st.session_state.use_manual_origin and st.session_state.use_manual_dest:
                # Both manual - swap all values
                st.session_state.manual_origin_name, st.session_state.manual_dest_name = st.session_state.manual_dest_name, st.session_state.manual_origin_name
                st.session_state.manual_origin_lat, st.session_state.manual_dest_lat = st.session_state.manual_dest_lat, st.session_state.manual_origin_lat
                st.session_state.manual_origin_lon, st.session_state.manual_dest_lon = st.session_state.manual_dest_lon, st.session_state.manual_origin_lon
            elif not st.session_state.use_manual_origin and not st.session_state.use_manual_dest:
                # Both from CSV - swap normally
                swap_ports()
            else:
                st.warning("⚠️ Cannot swap between manual and CSV ports")
                time.sleep(1)
            st.rerun()
    
    # Destination Port Selection
    st.subheader("🔴 Destination Port")
    use_manual_dest = st.checkbox("Enter manually", key="manual_dest_check", value=st.session_state.use_manual_dest)
    st.session_state.use_manual_dest = use_manual_dest
    
    if use_manual_dest:
        st.session_state.manual_dest_name = st.text_input("Port Name", value=st.session_state.manual_dest_name, key="manual_dest_name_input", placeholder="e.g., Custom Port")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.session_state.manual_dest_lat = st.number_input("Latitude", value=st.session_state.manual_dest_lat, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f", key="manual_dest_lat_input")
        with col_d2:
            st.session_state.manual_dest_lon = st.number_input("Longitude", value=st.session_state.manual_dest_lon, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f", key="manual_dest_lon_input")
    else:
        if not st.session_state.ports_df.empty:
            port_names = sorted(st.session_state.ports_df["port_name"].dropna().unique())
            dest_port = st.selectbox(
                "Select from list",
                port_names,
                index=port_names.index(st.session_state.dest_port) if st.session_state.dest_port in port_names else 1,
                key="dest_select"
            )
            st.session_state.dest_port = dest_port
        else:
            st.warning("⚠️ No port data available")
            st.session_state.dest_port = st.text_input("Destination Port", st.session_state.dest_port)
    
    speed_kts = st.slider(
        "⚡ Vessel Speed (knots)",
        min_value=1.0,
        max_value=30.0,
        value=st.session_state.speed_kts,
        step=0.5,
        help="Typical vessel speeds:\n• Cargo ships: 12-15 knots\n• Container ships: 20-25 knots\n• Tankers: 12-16 knots"
    )
    st.session_state.speed_kts = speed_kts
    
    st.markdown("")
    
    if st.button("🧭 Calculate Route", type="primary", use_container_width=True):
        # Validation for manual inputs
        if st.session_state.use_manual_origin:
            if not st.session_state.manual_origin_name or st.session_state.manual_origin_name.strip() == "":
                st.error("❌ Please enter an origin port name")
            elif st.session_state.manual_origin_lat == 0.0 and st.session_state.manual_origin_lon == 0.0:
                st.error("❌ Please enter valid origin coordinates")
            elif st.session_state.use_manual_dest:
                if not st.session_state.manual_dest_name or st.session_state.manual_dest_name.strip() == "":
                    st.error("❌ Please enter a destination port name")
                elif st.session_state.manual_dest_lat == 0.0 and st.session_state.manual_dest_lon == 0.0:
                    st.error("❌ Please enter valid destination coordinates")
                elif (st.session_state.manual_origin_lat == st.session_state.manual_dest_lat and 
                      st.session_state.manual_origin_lon == st.session_state.manual_dest_lon):
                    st.error("❌ Origin and destination cannot be the same")
                else:
                    calculate_route()
            else:
                calculate_route()
        elif st.session_state.use_manual_dest:
            if not st.session_state.manual_dest_name or st.session_state.manual_dest_name.strip() == "":
                st.error("❌ Please enter a destination port name")
            elif st.session_state.manual_dest_lat == 0.0 and st.session_state.manual_dest_lon == 0.0:
                st.error("❌ Please enter valid destination coordinates")
            else:
                calculate_route()
        else:
            if st.session_state.origin_port == st.session_state.dest_port:
                st.error("❌ Origin and destination must be different ports")
            else:
                calculate_route()
    
    st.markdown("---")
    st.info("💡 **How it works:**\n\n🟢 **SeaRoute** = Actual navigable maritime route (avoids land)\n\n🔴 **Haversine** = Theoretical shortest path (may go through land)")

# ---------- Main content ----------
if st.session_state.calculation_done and hasattr(st.session_state, 'map_obj'):
    
    # Port validation display
    if hasattr(st.session_state, 'origin_match_type') and hasattr(st.session_state, 'dest_match_type'):
        val_col1, val_col2 = st.columns(2)
        
        with val_col1:
            if st.session_state.origin_match_type == "exact":
                country_flag = f" 🌍 {st.session_state.origin_country}" if hasattr(st.session_state, 'origin_country') and st.session_state.origin_country != "Unknown" else ""
                st.markdown(f"✅ **Origin:** <span class='port-match-exact'>{st.session_state.origin_resolved}</span>{country_flag} (exact match)", unsafe_allow_html=True)
            elif st.session_state.origin_match_type in ["fuzzy", "partial"]:
                country_flag = f" 🌍 {st.session_state.origin_country}" if hasattr(st.session_state, 'origin_country') and st.session_state.origin_country != "Unknown" else ""
                st.markdown(f"⚠️ **Origin:** <span class='port-match-fuzzy'>{st.session_state.origin_resolved}</span>{country_flag} (fuzzy match)", unsafe_allow_html=True)
            elif st.session_state.origin_match_type == "manual":
                country_flag = f" 🌍 {st.session_state.origin_country}" if hasattr(st.session_state, 'origin_country') and st.session_state.origin_country != "Unknown" else ""
                st.markdown(f"✏️ **Origin:** <span class='port-match-exact'>{st.session_state.origin_resolved}</span>{country_flag} (manual entry)", unsafe_allow_html=True)
        
        with val_col2:
            if st.session_state.dest_match_type == "exact":
                country_flag = f" 🌍 {st.session_state.dest_country}" if hasattr(st.session_state, 'dest_country') and st.session_state.dest_country != "Unknown" else ""
                st.markdown(f"✅ **Destination:** <span class='port-match-exact'>{st.session_state.dest_resolved}</span>{country_flag} (exact match)", unsafe_allow_html=True)
            elif st.session_state.dest_match_type in ["fuzzy", "partial"]:
                country_flag = f" 🌍 {st.session_state.dest_country}" if hasattr(st.session_state, 'dest_country') and st.session_state.dest_country != "Unknown" else ""
                st.markdown(f"⚠️ **Destination:** <span class='port-match-fuzzy'>{st.session_state.dest_resolved}</span>{country_flag} (fuzzy match)", unsafe_allow_html=True)
            elif st.session_state.dest_match_type == "manual":
                country_flag = f" 🌍 {st.session_state.dest_country}" if hasattr(st.session_state, 'dest_country') and st.session_state.dest_country != "Unknown" else ""
                st.markdown(f"✏️ **Destination:** <span class='port-match-exact'>{st.session_state.dest_resolved}</span>{country_flag} (manual entry)", unsafe_allow_html=True)
    
    # Show SeaRoute error if exists
    if hasattr(st.session_state, 'sr_error') and st.session_state.sr_error:
        st.warning(st.session_state.sr_error)
    
    # Header with route info
    st.markdown(f"### 📍 {st.session_state.origin_resolved if hasattr(st.session_state, 'origin_resolved') else st.session_state.origin_port} → {st.session_state.dest_resolved if hasattr(st.session_state, 'dest_resolved') else st.session_state.dest_port}")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.sr_success and st.session_state.sr_dist:
            delta = f"+{st.session_state.sr_dist - st.session_state.haversine_dist:.1f} NM"
            st.metric(
                "🚢 SeaRoute Distance", 
                f"{st.session_state.sr_dist:,.1f} NM", 
                delta,
                help="Actual navigable maritime route distance that avoids land masses"
            )
        else:
            st.metric("🚢 SeaRoute", "N/A", help="SeaRoute calculation failed - see warning above")
    
    with col2:
        st.metric(
            "📏 Haversine Distance", 
            f"{st.session_state.haversine_dist:,.1f} NM",
            help="Theoretical shortest path (great circle distance) - may pass through land"
        )
    
    with col3:
        if st.session_state.sr_success and st.session_state.sr_dur:
            days = int(st.session_state.sr_dur // 24)
            hours = int(st.session_state.sr_dur % 24)
            duration_text = f"{days}d {hours}h" if days > 0 else f"{hours}h"
            st.metric(
                "⏱️ SeaRoute ETA", 
                duration_text,
                help=f"Estimated sailing time at {st.session_state.speed_kts} knots following the navigable route"
            )
        else:
            st.metric("⏱️ SeaRoute ETA", "N/A", help="SeaRoute calculation failed")
    
    with col4:
        h_days = int(st.session_state.haversine_dur // 24)
        h_hours = int(st.session_state.haversine_dur % 24)
        h_duration_text = f"{h_days}d {h_hours}h" if h_days > 0 else f"{h_hours}h"
        st.metric(
            "⏱️ Haversine ETA", 
            h_duration_text,
            help=f"Theoretical sailing time at {st.session_state.speed_kts} knots (assumes direct path)"
        )
    
    # Map
    st.markdown("---")
    st_folium(
        st.session_state.map_obj,
        key=f"folium_map_{st.session_state.map_key}",
        height=500,
        returned_objects=[],
        use_container_width=True
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Comparison & Methodology", "📋 Detailed Data", "💾 Export"])
    
    with tab1:
        if st.session_state.sr_success and st.session_state.sr_dist:
            col1, col2 = st.columns(2)
            
            fig_dist, fig_dur = create_comparison_chart(
                st.session_state.sr_dist,
                st.session_state.haversine_dist,
                st.session_state.sr_dur,
                st.session_state.haversine_dur
            )
            
            with col1:
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_dur, use_container_width=True)
            
            # Difference metrics with help tooltips
            st.markdown("### 📈 Route Comparison Metrics")
            diff_col1, diff_col2, diff_col3 = st.columns(3)
            
            with diff_col1:
                dist_diff = st.session_state.sr_dist - st.session_state.haversine_dist
                dist_pct = (dist_diff / st.session_state.haversine_dist) * 100
                st.metric(
                    "Distance Difference", 
                    f"{dist_diff:,.1f} NM", 
                    f"{dist_pct:.1f}%",
                    help="Extra distance the navigable route requires vs. theoretical shortest path\n\nFormula: SeaRoute Distance - Haversine Distance\n\nExample: If SeaRoute = 1500 NM and Haversine = 1200 NM, then difference = +300 NM (+25%)"
                )
            
            with diff_col2:
                dur_diff = st.session_state.sr_dur - st.session_state.haversine_dur
                st.metric(
                    "Time Difference", 
                    f"{dur_diff:.1f}h",
                    help=f"Extra sailing time needed for navigable route (both calculated at {st.session_state.speed_kts} knots)\n\nFormula: SeaRoute Duration - Haversine Duration\n\nNote: Haversine time is theoretical since you cannot sail through land!"
                )
            
            with diff_col3:
                efficiency = (st.session_state.haversine_dist / st.session_state.sr_dist) * 100
                st.metric(
                    "Route Efficiency", 
                    f"{efficiency:.1f}%",
                    help="How direct the navigable route is compared to ideal straight line\n\nFormula: (Haversine Distance ÷ SeaRoute Distance) × 100%\n\n100% = Perfectly direct\n90% = 10% longer than ideal\n80% = 25% longer than ideal"
                )
        else:
            st.warning("⚠️ SeaRoute calculation not available. Only Haversine route is shown.")
        
        # Methodology explanation - ALWAYS VISIBLE
        st.markdown("---")
        with st.expander("📚 **How are these calculations done? Click to learn more!**", expanded=False):
            st.markdown("""
            ## 🧭 Route Calculation Methods
            
            ### 🟢 SeaRoute (Navigable Maritime Route)
            
            **What it does:**
            - Calculates the actual path a ship would take in real navigation
            - Considers maritime navigation constraints
            - Avoids land masses, shallow waters, and restricted zones
            - Uses established shipping lanes for safety
            - May route through straits and canals (Suez, Panama, Malacca)
            
            **How it's calculated:**
            ```python
            # Using searoute Python library
            import searoute as sr
            
            route = sr.searoute(
                origin=[longitude, latitude],
                destination=[longitude, latitude],
                units="naut",        # Nautical miles
                speed_knot=12        # Your vessel speed
            )
            
            distance = route['properties']['length']
            duration = route['properties']['duration_hours']
            ```
            
            ---
            
            ### 🔴 Haversine (Great Circle Route)
            
            **What it does:**
            - Calculates shortest distance between two points on Earth's surface
            - Does NOT consider any obstacles or navigation constraints
            - This is **theoretical only** - you can't actually sail this if land is in the way!
            - Useful as a **baseline** to measure detour requirements
            
            **The Haversine Formula:**
            ```
            Step 1: Convert to radians
            φ₁ = lat₁ × (π/180)
            φ₂ = lat₂ × (π/180)
            λ₁ = lon₁ × (π/180)
            λ₂ = lon₂ × (π/180)
            
            Step 2: Calculate differences
            Δφ = φ₂ - φ₁
            Δλ = λ₂ - λ₁
            
            Step 3: Haversine formula
            a = sin²(Δφ/2) + cos(φ₁) × cos(φ₂) × sin²(Δλ/2)
            c = 2 × arcsin(√a)
            
            Step 4: Distance
            distance_km = Earth_radius × c
            distance_NM = distance_km ÷ 1.852
            ```
            
            **Duration formula:**
            ```
            Duration (hours) = Distance (NM) ÷ Speed (knots)
            ```
            Note: 1 knot = 1 nautical mile per hour
            
            ---
            
            ## 📊 Understanding the Comparison Metrics
            
            ### 1️⃣ Distance Difference
            
            **Formula:**
            ```
            Difference (NM) = SeaRoute Distance - Haversine Distance
            Percentage (%) = (Difference ÷ Haversine Distance) × 100
            ```
            
            **What it means:** Shows how much extra distance the navigable route requires
            
            **Example:**
            - SeaRoute: 1,500 NM (actual route avoiding land)
            - Haversine: 1,200 NM (straight line, may cut through continents)
            - **Difference: +300 NM (+25%)**
            
            ---
            
            ### 2️⃣ Time Difference
            
            **Formula:**
            ```
            Time Difference (hours) = SeaRoute Duration - Haversine Duration
            ```
            
            **What it means:** Extra sailing time needed for the navigable route
            
            **Example:**
            - SeaRoute: 125 hours (at 12 knots)
            - Haversine: 100 hours (at 12 knots)
            - **Difference: +25 hours**
            
            ⚠️ **Important:** Both calculations use the same speed, so time difference is proportional to distance. The Haversine time is theoretical only!
            
            ---
            
            ### 3️⃣ Route Efficiency
            
            **Formula:**
            ```
            Efficiency (%) = (Haversine Distance ÷ SeaRoute Distance) × 100
            ```
            
            **What it means:** How direct the navigable route is
            
            **Interpretation:**
            - **100%** = Perfectly direct (no obstacles)
            - **90%** = 11% longer than ideal
            - **80%** = 25% longer than ideal
            - **70%** = 43% longer (major detours around continents)
            
            Lower efficiency means more geographical constraints!
            
            ---
            
            ## 🌍 Why is SeaRoute Always Longer?
            
            The navigable route must:
            
            1. **Navigate around land** (continents, islands)
            2. **Use safe shipping lanes**
            3. **Pass through chokepoints:**
               - Suez Canal (Asia ↔ Europe)
               - Panama Canal (Pacific ↔ Atlantic)
               - Strait of Malacca
               - Strait of Gibraltar
            4. **Avoid hazards:**
               - Shallow waters
               - Piracy zones
               - Restricted military areas
               - Dangerous weather regions
            
            The Haversine route ignores all this - it's just the shortest path on a sphere, often cutting through land!
            
            ---
            
            ## 🔢 Units Reference
            
            **Distance:** Nautical Miles (NM)
            - 1 NM = 1.852 kilometers
            - 1 NM = 1.15078 statute miles
            
            **Speed:** Knots (NM per hour)
            - 1 knot = 1 NM/hour
            - 12 knots ≈ 22 km/h ≈ 14 mph
            
            **Time:** Hours
            - Convertible to days (÷ 24)
            """)
    
    with tab2:
        st.markdown("### 📍 Route Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Origin Port**")
            if st.session_state.use_manual_origin:
                st.write(f"• Name: {st.session_state.manual_origin_name}")
                st.write(f"• Country: {st.session_state.origin_country if hasattr(st.session_state, 'origin_country') else 'Unknown'}")
                st.write(f"• Coordinates: {st.session_state.manual_origin_lat:.4f}°, {st.session_state.manual_origin_lon:.4f}°")
            else:
                olat, olon, _, _, ocountry = get_coords(st.session_state.origin_port)
                if olat and olon:
                    st.write(f"• Name: {st.session_state.origin_resolved if hasattr(st.session_state, 'origin_resolved') else st.session_state.origin_port}")
                    st.write(f"• Country: {st.session_state.origin_country if hasattr(st.session_state, 'origin_country') else ocountry}")
                    st.write(f"• Coordinates: {olat:.4f}°, {olon:.4f}°")
        
        with col2:
            st.markdown("**Destination Port**")
            if st.session_state.use_manual_dest:
                st.write(f"• Name: {st.session_state.manual_dest_name}")
                st.write(f"• Country: {st.session_state.dest_country if hasattr(st.session_state, 'dest_country') else 'Unknown'}")
                st.write(f"• Coordinates: {st.session_state.manual_dest_lat:.4f}°, {st.session_state.manual_dest_lon:.4f}°")
            else:
                dlat, dlon, _, _, dcountry = get_coords(st.session_state.dest_port)
                if dlat and dlon:
                    st.write(f"• Name: {st.session_state.dest_resolved if hasattr(st.session_state, 'dest_resolved') else st.session_state.dest_port}")
                    st.write(f"• Country: {st.session_state.dest_country if hasattr(st.session_state, 'dest_country') else dcountry}")
                    st.write(f"• Coordinates: {dlat:.4f}°, {dlon:.4f}°")
        
        st.markdown("---")
        
        # Comparison table
        comparison_data = {
            "Metric": ["Distance (NM)", "Duration (hours)", "Duration (days)", "Average Speed (knots)"],
            "SeaRoute": [
                f"{st.session_state.sr_dist:,.1f}" if st.session_state.sr_success else "N/A",
                f"{st.session_state.sr_dur:.1f}" if st.session_state.sr_success else "N/A",
                f"{st.session_state.sr_dur / 24:.1f}" if st.session_state.sr_success else "N/A",
                f"{st.session_state.speed_kts:.1f}"
            ],
            "Haversine": [
                f"{st.session_state.haversine_dist:,.1f}",
                f"{st.session_state.haversine_dur:.1f}",
                f"{st.session_state.haversine_dur / 24:.1f}",
                f"{st.session_state.speed_kts:.1f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
    
    with tab3:
        st.markdown("### 💾 Export Route Data")
        
        # Calculate route efficiency safely
        route_efficiency = "N/A"
        if st.session_state.sr_success and st.session_state.sr_dist and st.session_state.sr_dist > 0:
            route_efficiency = f"{(st.session_state.haversine_dist / st.session_state.sr_dist * 100):.2f}"
        
        export_data = pd.DataFrame({
            "Parameter": [
                "Origin Port",
                "Destination Port",
                "Vessel Speed (knots)",
                "SeaRoute Distance (NM)",
                "Haversine Distance (NM)",
                "SeaRoute Duration (hours)",
                "Haversine Duration (hours)",
                "Distance Difference (NM)",
                "Time Difference (hours)",
                "Route Efficiency (%)"
            ],
            "Value": [
                st.session_state.origin_resolved if hasattr(st.session_state, 'origin_resolved') else st.session_state.origin_port,
                st.session_state.dest_resolved if hasattr(st.session_state, 'dest_resolved') else st.session_state.dest_port,
                st.session_state.speed_kts,
                st.session_state.sr_dist if st.session_state.sr_success else "N/A",
                st.session_state.haversine_dist,
                st.session_state.sr_dur if st.session_state.sr_success else "N/A",
                st.session_state.haversine_dur,
                st.session_state.sr_dist - st.session_state.haversine_dist if st.session_state.sr_success else "N/A",
                st.session_state.sr_dur - st.session_state.haversine_dur if st.session_state.sr_success else "N/A",
                route_efficiency
            ]
        })
        
        csv = export_data.to_csv(index=False)
        st.download_button(
            "📥 Download Route Data (CSV)",
            data=csv,
            file_name=f"route_{st.session_state.origin_port}_{st.session_state.dest_port}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👈 **Get Started:** Select your origin and destination ports from the sidebar, then click **Calculate Route**")
    
    # Welcome map
    welcome_map = folium.Map(
        location=[-2.5, 118],
        zoom_start=4,
        tiles="CartoDB positron"
    )
    st_folium(welcome_map, key="welcome_map", height=500, returned_objects=[], use_container_width=True)

st.markdown("---")
st.caption("🌊 Maritime Route Comparison Tool • SeaRoute vs Haversine • Powered by Folium & Plotly")
