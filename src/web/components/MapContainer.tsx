"use client";

import { MapContainer, TileLayer, Marker, Polygon, useMapEvents, useMap } from "react-leaflet";
import { useEffect } from "react";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Fix Leaflet's default icon path issues in Next.js
const icon = L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
});

interface MapProps {
    onLocationSelect: (lat: number, lng: number) => void;
    selectedLocation: { lat: number; lng: number } | null;
    cellBounds?: [number, number][] | null;
    spreadResult?: any | null;
}

function LocationMarker({ onLocationSelect, selectedLocation }: MapProps) {
    useMapEvents({
        click(e) {
            onLocationSelect(e.latlng.lat, e.latlng.lng);
        },
    });

    return selectedLocation === null ? null : (
        <Marker position={[selectedLocation.lat, selectedLocation.lng]} icon={icon} />
    );
}

function RecenterAutomatically({ lat, lng }: { lat: number; lng: number }) {
    const map = useMap();
    useEffect(() => {
        map.flyTo([lat, lng], map.getZoom(), { duration: 1.5 });
    }, [lat, lng, map]);
    return null;
}

export default function IberFireMap({ onLocationSelect, selectedLocation, cellBounds, spreadResult }: MapProps) {
    return (
        <MapContainer
            center={[40.4168, -3.7038]} // Center of Spain
            zoom={6}
            minZoom={5}
            maxBounds={[
                [35.0, -10.0], // South-West (Canary Islands exluded usually, but covering peninsula)
                [44.0, 5.0]    // North-East
            ]}
            maxBoundsViscosity={1.0}
            scrollWheelZoom={true}
            className="h-full w-full z-0 relative"
        >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {/* TODO: Add Topographic or Satellite layer toggles here */}

            {selectedLocation && (
                <RecenterAutomatically lat={selectedLocation.lat} lng={selectedLocation.lng} />
            )}
            <LocationMarker onLocationSelect={onLocationSelect} selectedLocation={selectedLocation} />

            {/* Draw grid cell if available */}
            {cellBounds && cellBounds.length > 0 && (
                <Polygon
                    positions={cellBounds}
                    pathOptions={{ color: 'red', fillColor: 'orange', fillOpacity: 0.2, weight: 2 }}
                />
            )}

            {/* Draw spread polygons */}
            {spreadResult && spreadResult.geojson_polygon && spreadResult.geojson_polygon.map((poly: number[][], idx: number) => (
                <Polygon
                    key={`spread_${idx}`}
                    positions={poly.map(p => [p[1], p[0]] as [number, number])}
                    pathOptions={{ color: '#ff0000', fillColor: '#ff4400', fillOpacity: 0.6, weight: 2, dashArray: '4' }}
                />
            ))}
        </MapContainer>
    );
}
