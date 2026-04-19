"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Sidebar, { IgnitionResult, Features } from "@/components/Sidebar";
import axios from "axios";

// Dynamically import MapContainer with no SSR to prevent Leaflet window errors
const MapContainer = dynamic(() => import("@/components/MapContainer"), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full flex items-center justify-center bg-gray-100">
            <div className="animate-pulse text-gray-500">Cargando mapa...</div>
        </div>
    ),
});

interface Location {
    lat: number;
    lng: number;
}

export default function Home() {
    const [globalDate, setGlobalDate] = useState(new Date().toISOString().split("T")[0]);
    const [selectedLocation, setSelectedLocation] = useState<Location | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [ignitionResult, setIgnitionResult] = useState<IgnitionResult | null>(null);
    const [features, setFeatures] = useState<Features | null>(null);
    const [cellBounds, setCellBounds] = useState<[number, number][] | null>(null);
    
    // Spread State
    const [isSpreadLoading, setIsSpreadLoading] = useState(false);
    const [spreadResult, setSpreadResult] = useState<any | null>(null);



    const handleDateChange = (newDate: string) => {
        setGlobalDate(newDate);
        if (selectedLocation) {
            handleLocationSelect(selectedLocation.lat, selectedLocation.lng, newDate);
        }
    };

    const handleLocationSelect = async (lat: number, lng: number, overrideDate?: string) => {
        setSelectedLocation({ lat, lng });
        setIgnitionResult(null); // Reset previous result
        setFeatures(null);
        setCellBounds(null);
        setSpreadResult(null); // Reset spread when clicking new point
        setIsLoading(true);

        try {
            console.log(`[FRONTEND] -> Enviando click al Backend`);
            console.log(`[FRONTEND] -> Lat: ${lat}, Lon: ${lng}, Fecha elegida: ${overrideDate || globalDate}`);

            // API call to local FastAPI backend for ignition risk
            const response = await axios.post("http://127.0.0.1:8000/predict/ignition", {
                lat,
                lon: lng,
                date: overrideDate || globalDate,
            });
            setIgnitionResult(response.data);
            if (response.data.features) setFeatures(response.data.features);
            if (response.data.cell_bounds) setCellBounds(response.data.cell_bounds);
        } catch (error) {
            console.error("Error fetching ignition risk:", error);
            // Fallback dummy for UI testing if backend is down
            setIgnitionResult({
                probability: 0.12,
                risk_level: "Low",
                explanation: "API Unreachable. Mock result.",
            });
        } finally {
            setIsLoading(false);
        }
    };

    const handleSimulateSpread = async () => {
        if (!selectedLocation || !globalDate) return;
        setIsSpreadLoading(true);
        setSpreadResult(null);

        try {
            console.log(`[FRONTEND] -> Solicitando simulación de propagación`);
            const response = await axios.post("http://127.0.0.1:8000/predict/spread", {
                lat: selectedLocation.lat,
                lon: selectedLocation.lng,
                date: globalDate,
                duration_hours: 48,
            });
            setSpreadResult(response.data);
        } catch (error) {
            console.error("Error fetching spread prediction:", error);
        } finally {
            setIsSpreadLoading(false);
        }
    };



    return (
        <main className="flex h-screen w-full overflow-hidden bg-gray-50">
            <Sidebar
                globalDate={globalDate}
                onDateChange={handleDateChange}
                selectedLocation={selectedLocation}
                onLocationInput={handleLocationSelect}
                isLoading={isLoading}
                ignitionResult={ignitionResult}
                features={features}
                onSimulateSpread={handleSimulateSpread}
                isSpreadLoading={isSpreadLoading}
                spreadResult={spreadResult}
            />
            <div className="flex-1 relative h-full">
                <MapContainer
                    onLocationSelect={handleLocationSelect}
                    selectedLocation={selectedLocation}
                    cellBounds={cellBounds}
                    spreadResult={spreadResult}
                />
            </div>
        </main>
    );
}
