import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PlotComponent = ({ plotType, showForecast, handleForecast }) => {
    const [plotUrl, setPlotUrl] = useState('');
    const [rmse, setRmse] = useState('');
    const [plotTitle, setPlotTitle] = useState('');

    useEffect(() => {
       
        let url;
        let title;
        if (plotType === 'plot') {
            url = 'http://localhost:4000/plot';
            title = 'Plot';
        } else if (plotType === 'stationary') {
            url = 'http://localhost:4000/stationary_plot';
            title = 'Stationary Plot';
        } else if (plotType === 'forecast' || plotType === 'ann_forecast' || plotType === 'arima_forecast' || plotType === 'sarima_forecast' || plotType === 'ets_forecast' || plotType === 'svr_forecast' || plotType === 'lstm_forecast' || plotType === 'hybrid_forecast' || plotType === 'prophet_forecast') {
            url = 'http://localhost:4000/' + plotType;
            title = plotType;
        } else if (plotType === 'ann') {
            url = 'http://localhost:4000/ann_plot';
            title = 'ANN Plot';
        } else if (plotType === 'arima') {
            url = 'http://localhost:4000/arima_plot';
            title = 'ARIMA Plot';
        } else if (plotType === 'sarima') {
            url = 'http://localhost:4000/sarima_plot';
            title = 'SARIMA Plot';
        } else if (plotType === 'ets') {
            url = 'http://localhost:4000/ets_plot';
            title = 'ETS Plot';
        } else if (plotType === 'svr') {
            url = 'http://localhost:4000/svr_plot';
            title = 'SVR Plot';
        } else if (plotType === 'lstm') {
            url = 'http://localhost:4000/lstm_plot';
            title = 'LSTM Plot';
        } else if (plotType === 'hybrid') {
            url = 'http://localhost:4000/hybrid_plot';
            title = 'HYBRID Plot';
        } else if (plotType === 'prophet') {
            url = 'http://localhost:4000/prophet_plot';
            title = 'Prophet Plot';
        }
        axios.get(url, { responseType: 'blob' }) 
            .then(response => {
                const imageUrl = URL.createObjectURL(response.data); 
                setPlotUrl(imageUrl);
                setPlotTitle(title);
            })
            .catch(error => {
                console.error('Error fetching plot:', error);
            });

        if (plotType === 'arima') {
            axios.get(url)
                .then(response => {
                    setRmse(response.headers['RMSE']);
                })
                .catch(error => {
                    console.error('Error fetching RMSE:', error);
                });
        }
    }, [plotType, showForecast]);

    return (
        <div style={{ flex: 1 }}>
            <h1>{plotTitle}</h1>
            {plotUrl && <img src={plotUrl} alt={plotTitle} style={{ maxWidth: '100%' }} />}
            {plotType === 'arima' && rmse && <p>RMSE: {rmse}</p>}
            {showForecast && <button style={{ backgroundColor: 'black', color: 'white' }} onClick={handleForecast}>Forecast</button>}
        </div>
    );
};


const PlotSelector = () => {
    const [selectedPlot, setSelectedPlot] = useState('plot');
    const [selectedModel, setSelectedModel] = useState('arima');
    const [showForecast, setShowForecast] = useState(false);

    const handlePlotChange = (plotType) => {
        setSelectedPlot(plotType);
    };

    const handleModelChange = (modelType) => {
        setSelectedModel(modelType);
        setSelectedPlot(modelType);
    };

    const handleSubmit = () => {
        setSelectedPlot(selectedModel);
        setShowForecast(true);
    };

    const handleForecast = () => {
        if (selectedModel === 'ann') {
            setSelectedPlot('ann_forecast');
        } else if (selectedModel === 'arima') {
            setSelectedPlot('arima_forecast');
        } else if (selectedModel === 'sarima') {
            setSelectedPlot('sarima_forecast');
        } else if (selectedModel === 'ets') {
            setSelectedPlot('ets_forecast');
        } else if (selectedModel === 'svr') {
            setSelectedPlot('svr_forecast');
        } else if (selectedModel === 'lstm') {
            setSelectedPlot('lstm_forecast');
        } else if (selectedModel === 'hybrid') {
            setSelectedPlot('hybrid_forecast');
        } else if (selectedModel === 'prophet') {
            setSelectedPlot('prophet_forecast');
        } else {
            setSelectedPlot('forecast');
        }
        setShowForecast(false);
    };
    
    return (
        <div style={{ background: 'linear-gradient(180deg, #000000 0%, #0000FF 100%)', minHeight: '100vh', color: '#FFFFFF' }}>
            <div style={{ display: 'flex', padding: '20px' }}>
            <div style={{ display: 'flex' }}>
                <div style={{ flex: 1 }}>
                    <h1>Select Plot</h1>
                    <button style={{ backgroundColor: 'black', color: 'white' }} onClick={() => handlePlotChange('plot')}>Show Plot</button>
                    <button style={{ backgroundColor: 'black', color: 'white' }} onClick={() => handlePlotChange('stationary')}>Show Stationary Plot</button>
                    <h1>Select Model</h1>
                    <select value={selectedModel} onChange={(e) => handleModelChange(e.target.value)}>
                        <option value="arima">ARIMA</option>
                        <option value="ann">ANN</option>
                        <option value="sarima">SARIMA</option>
                        <option value="svr">SVR</option>
                        <option value="ets">ETS</option>
                        <option value="prophet">Prophet</option>
                        <option value="lstm">LSTM</option>
                        <option value="hybrid">HYBRID</option>
                    </select>
                    <button style={{ backgroundColor: 'black', color: 'white' }} onClick={handleSubmit}>Submit</button>
                </div>
                <div style={{ flex: 1 }}>
                    <PlotComponent plotType={selectedPlot} showForecast={showForecast} handleForecast={handleForecast} />
                </div>
            </div>
        </div>
    </div>
    );
};

export default PlotSelector;
