"""Render a selected trip's model output as a Folium map."""

import folium


def render_trip_map(result):
    gps = result["gps_points"]  # [longitude, latitude]
    if not gps:
        raise ValueError("Trip contains no GPS observations")

    center = [sum(p[1] for p in gps) / len(gps), sum(p[0] for p in gps) / len(gps)]
    fmap = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", control_scale=True)

    gps_layer = folium.FeatureGroup(name="Original GPS trajectory", show=True)
    folium.PolyLine(
        [(lat, lon) for lon, lat in gps], color="#498cdb", weight=3,
        opacity=0.8, tooltip="Original GPS trajectory"
    ).add_to(gps_layer)
    folium.CircleMarker(
        location=[gps[0][1], gps[0][0]], radius=7, color="white",
        weight=2, fill=True, fill_color="#29b894", fill_opacity=1,
        tooltip="GPS start"
    ).add_to(gps_layer)
    folium.CircleMarker(
        location=[gps[-1][1], gps[-1][0]], radius=7, color="white",
        weight=2, fill=True, fill_color="#dc5b53", fill_opacity=1,
        tooltip="GPS end"
    ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # GeoJSON MultiLineString keeps the map compact even for long trips.
    unique_segments = dict(zip(result["predicted_arc_ids"], result["predicted_segments"]))
    route_layer = folium.FeatureGroup(name="HMM / Viterbi predicted route", show=True)
    folium.GeoJson(
        {"type": "Feature", "properties": {}, "geometry": {
            "type": "MultiLineString", "coordinates": list(unique_segments.values())
        }},
        style_function=lambda _: {"color": "#ed6b61", "weight": 4, "opacity": 0.88},
        tooltip="Predicted road segment"
    ).add_to(route_layer)
    route_layer.add_to(fmap)

    points = gps + [p for segment in unique_segments.values() for p in segment]
    fmap.fit_bounds([
        [min(p[1] for p in points), min(p[0] for p in points)],
        [max(p[1] for p in points), max(p[0] for p in points)],
    ])
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap.get_root().render()
