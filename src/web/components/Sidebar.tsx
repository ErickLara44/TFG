import { useState } from "react";
import { Loader2, Flame } from "lucide-react";

export interface IgnitionResult {
    risk_level?: string;
    probability?: number;
    explanation?: string;
}

export interface Features {
    temperature?: number;
    humidity?: number;
    wind_speed?: number;
    wind_direction?: number;
    elevation?: number;
    forest_prop?: number;
    precipitation?: number;
    FWI?: number;
}

interface SidebarProps {
    globalDate: string;
    onDateChange: (newDate: string) => void;
    selectedLocation: { lat: number; lng: number } | null;
    onLocationInput: (lat: number, lng: number) => void;
    isLoading: boolean;
    ignitionResult: IgnitionResult | null;
    features?: Features | null;
    onSimulateSpread: () => void;
    isSpreadLoading: boolean;
    spreadResult: any | null;
}

export default function Sidebar({ globalDate, onDateChange, selectedLocation, onLocationInput, isLoading, ignitionResult, features, onSimulateSpread, isSpreadLoading, spreadResult }: SidebarProps) {
    const [inputLat, setInputLat] = useState("");
    const [inputLng, setInputLng] = useState("");

    const handleGoToLocation = () => {
        const lat = parseFloat(inputLat);
        const lng = parseFloat(inputLng);
        if (!isNaN(lat) && !isNaN(lng)) {
            onLocationInput(lat, lng);
        }
    };

    // We no longer need local editable state since everything is read-only
    // from the automatically fetched `features` props.

    return (
        <div className="w-80 h-full bg-white shadow-xl flex flex-col z-10 p-6 overflow-y-auto">
            <h1 className="text-2xl font-bold mb-6 flex items-center text-orange-600">
                <Flame className="mr-2" />
                IberFire
            </h1>

            <div className="space-y-6">
                {/* Time Context Section */}
                <div>
                    <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">
                        Fecha de Predicción
                    </h2>
                    <input
                        type="date"
                        value={globalDate}
                        onChange={(e) => onDateChange(e.target.value)}
                        min="2008-01-01"
                        max={new Date().toISOString().split("T")[0]}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 text-sm font-medium text-gray-700 bg-gray-50"
                    />
                </div>

                {/* Location Section */}
                <div>
                    <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">
                        Localización / Coordenadas
                    </h2>
                    
                    <div className="flex space-x-2 mb-3">
                        <input
                            type="number"
                            placeholder="Latitud"
                            value={inputLat}
                            onChange={(e) => setInputLat(e.target.value)}
                            className="w-1/2 px-2 py-1.5 border border-gray-300 rounded-md text-sm focus:ring-1 focus:ring-orange-500"
                        />
                        <input
                            type="number"
                            placeholder="Longitud"
                            value={inputLng}
                            onChange={(e) => setInputLng(e.target.value)}
                            className="w-1/2 px-2 py-1.5 border border-gray-300 rounded-md text-sm focus:ring-1 focus:ring-orange-500"
                        />
                        <button
                            onClick={handleGoToLocation}
                            className="px-3 py-1.5 bg-orange-100 text-orange-700 font-medium text-sm rounded-md hover:bg-orange-200 transition-colors"
                        >
                            Ir
                        </button>
                    </div>

                    {selectedLocation ? (
                        <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                            <p className="text-sm">
                                Lat: <span className="font-mono font-medium">{selectedLocation.lat.toFixed(4)}</span>
                            </p>
                            <p className="text-sm">
                                Lon: <span className="font-mono font-medium">{selectedLocation.lng.toFixed(4)}</span>
                            </p>
                        </div>
                    ) : (
                        <p className="text-sm text-gray-500 italic bg-blue-50 p-3 rounded-lg border border-blue-100">
                            📍 Haz click en España o introduce lat/lon arriba para predecir.
                        </p>
                    )}
                </div>

                {/* Loading State */}
                {isLoading && (
                    <div className="flex flex-col items-center justify-center p-6 space-y-3 bg-orange-50 rounded-lg border border-orange-200">
                        <Loader2 className="animate-spin text-orange-600" size={32} />
                        <p className="text-sm font-medium text-orange-800 text-center">
                            Conectando con Open-Meteo y Copernicus...
                        </p>
                        <p className="text-xs text-orange-600/70 text-center">
                            Extrayendo relieve, vegetación y viento en tiempo real.
                        </p>
                    </div>
                )}

                {/* Results Section */}
                {!isLoading && ignitionResult && (
                    <div className="pt-2">
                        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
                            Riesgo de Ignición AI
                        </h2>
                        <div className={`p-4 rounded-lg flex items-center justify-between shadow-sm ${ignitionResult.risk_level === 'High' ? 'bg-red-50 text-red-700 border border-red-200' :
                            ignitionResult.risk_level === 'Moderate' ? 'bg-yellow-50 text-yellow-700 border border-yellow-200' :
                                ignitionResult.risk_level === 'Extreme' ? 'bg-purple-50 text-purple-700 border border-purple-200' :
                                    'bg-green-50 text-green-700 border border-green-200'
                            }`}>
                            <div>
                                <p className="font-bold text-lg">{ignitionResult.risk_level}</p>
                                <p className="text-xs opacity-80 mt-1">ConvLSTM 64x64</p>
                            </div>
                            <div className="text-3xl font-black">
                                {Math.round((ignitionResult.probability || 0) * 100)}%
                            </div>
                        </div>
                        
                        {/* Area for Spread Prediction Button */}
                        <div className="mt-4">
                            <button
                                onClick={onSimulateSpread}
                                disabled={isSpreadLoading}
                                className="w-full flex items-center justify-center py-2 px-4 rounded-lg bg-gradient-to-r from-orange-500 to-red-600 text-white font-bold shadow-md hover:from-orange-600 hover:to-red-700 disabled:opacity-50 transition-all transition-transform active:scale-95"
                            >
                                {isSpreadLoading ? (
                                    <>
                                        <Loader2 className="animate-spin mr-2" size={18} />
                                        Simulando Avance (48h)...
                                    </>
                                ) : (
                                    <>
                                        🔥 Simular Propagación
                                    </>
                                )}
                            </button>
                        </div>
                        
                        {spreadResult && spreadResult.area_hectares && (
                            <div className="mt-3 bg-red-50 p-3 rounded-lg border border-red-200">
                                <p className="text-sm text-red-800 font-semibold">Área Quemada Estimada:</p>
                                <p className="text-xl font-black text-red-600">{spreadResult.area_hectares} ha</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Read-Only Features Section */}
                {!isLoading && features && (
                    <div className="pt-2 border-t border-gray-100">
                        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">
                            Condiciones Extraídas (Auto)
                        </h2>
                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 grid grid-cols-2 gap-y-3 gap-x-2">
                            <div>
                                <p className="text-[10px] uppercase font-bold text-gray-400">Temperatura</p>
                                <p className="text-sm font-medium text-gray-700">{features.temperature?.toFixed(1)} °C</p>
                            </div>
                            <div>
                                <p className="text-[10px] uppercase font-bold text-gray-400">Humedad</p>
                                <p className="text-sm font-medium text-gray-700">{features.humidity?.toFixed(1)} %</p>
                            </div>
                            <div>
                                <p className="text-[10px] uppercase font-bold text-gray-400">Viento</p>
                                <p className="text-sm font-medium text-gray-700">{((features.wind_speed || 0) * 3.6).toFixed(1)} km/h</p>
                            </div>
                            <div>
                                <p className="text-[10px] uppercase font-bold text-gray-400">Dir. Viento</p>
                                <p className="text-sm font-medium text-gray-700">{features.wind_direction?.toFixed(0)}°</p>
                            </div>
                            <div>
                                <p className="text-[10px] uppercase font-bold text-gray-400">Elevación</p>
                                <p className="text-sm font-medium text-gray-700">{features.elevation?.toFixed(0)} m</p>
                            </div>
                            <div>
                                <p className="text-[10px] uppercase font-bold text-gray-400">Bosque</p>
                                <p className="text-sm font-medium text-gray-700">{((features.forest_prop || 0) * 100).toFixed(1)} %</p>
                            </div>
                        </div>

                        {/* FWI computed badge */}
                        {features.FWI !== undefined && (
                            <div className="mt-3 flex items-center justify-between bg-orange-50 border border-orange-200 rounded-lg px-3 py-2 shadow-sm">
                                <div>
                                    <p className="text-xs text-orange-800 font-bold">FWI Index</p>
                                    <p className="text-[10px] text-orange-600">Calculado a 7 días</p>
                                </div>
                                <span className="text-lg font-black text-orange-700">{features.FWI?.toFixed(1)}</span>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
