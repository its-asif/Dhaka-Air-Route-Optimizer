document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initForecastChart();
    initXAICharts();
    loadLeaderboard();

    document.getElementById("btnForecast").addEventListener("click", runForecast);
    document.getElementById("btnRunBenchmark").addEventListener("click", runBenchmark);
    document.getElementById("btnCalculateHealthRoutes").addEventListener("click", calculateHealthRoutes);
});

// TAB SWITCHING
function initTabs() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetId = `tab-${tab.dataset.tab}`;
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add("active");
            }
        });
    });
}

// CHART OBJECT REFERENCES
let forecastChart, flLossChart, shapChart, igPieChart, attnChart;

function initForecastChart() {
    const ctx = document.getElementById("forecastChart").getContext("2d");
    const hours = Array.from({length: 24}, (_, i) => `+${i+1}h`);
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [{
                label: 'Forecasted AQI',
                data: Array.from({length: 24}, () => Math.floor(130 + Math.random() * 40)),
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
            }
        }
    });
}

function runForecast() {
    const model = document.getElementById("modelSelect").value;
    fetch("/research/api/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: model })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            document.getElementById("forecastAqiVal").innerText = data.predicted_aqi;
            document.getElementById("forecastStatus").innerText = data.status;

            // Update line chart data dynamically
            const base = data.predicted_aqi;
            const newData = Array.from({length: 24}, (_, i) => Math.round(base + Math.sin(i / 2) * 15 + Math.random() * 5));
            forecastChart.data.datasets[0].data = newData;
            forecastChart.update();
            loadXAI(model);
        }
    })
    .catch(err => console.error(err));
}



function loadLeaderboard() {
    fetch("/research/api/experiments/leaderboard/")
    .then(res => res.json())
    .then(data => {
        if (data.success && data.leaderboard_data) {
            renderLeaderboardTable(data.leaderboard_data);
        }
    })
    .catch(err => console.error(err));
}

function runBenchmark() {
    fetch("/research/api/experiments/leaderboard/", { method: "POST" })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            loadLeaderboard();
        }
    });
}

function renderLeaderboardTable(data) {
    const tbody = document.querySelector("#leaderboardTable tbody");
    tbody.innerHTML = "";

    document.getElementById("bestModelBanner").innerHTML = 
        `🏆 Best Model: <strong>${data.best_model} (${data.best_setup})</strong> — RMSE: <strong>${data.best_rmse}</strong> | R²: <strong>${data.best_r2}</strong>`;

    data.leaderboard.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${row.Model}</strong></td>
            <td><span class="badge ${row.Setup === 'Federated FedAvg' ? 'badge-success' : 'badge-warning'}">${row.Setup}</span></td>
            <td>${row.RMSE}</td>
            <td>${row.MAE}</td>
            <td>${row['MAPE (%)']}</td>
            <td><strong>${row['R2 Score']}</strong></td>
        `;
        tbody.appendChild(tr);
    });
}

function initXAICharts() {
    const ctx1 = document.getElementById("shapChart").getContext("2d");
    shapChart = new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'Temp', 'Humidity', 'Wind'],
            datasets: [{
                label: 'SHAP Value Impact',
                data: [42.5, 31.2, 12.4, 6.1, 4.2, 2.1, 1.5, -3.2],
                backgroundColor: ['#ef4444', '#f97316', '#f59e0b', '#3b82f6', '#8b5cf6', '#10b981', '#06b6d4', '#64748b']
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
            }
        }
    });

    const ctx2 = document.getElementById("igPieChart").getContext("2d");
    igPieChart = new Chart(ctx2, {
        type: 'doughnut',
        data: {
            labels: ['Pollutant Concentration Impact', 'Meteorological Factors Impact'],
            datasets: [{
                data: [82.4, 17.6],
                backgroundColor: ['#ef4444', '#38bdf8']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8' } } }
        }
    });

    const ctx3 = document.getElementById("attentionChart").getContext("2d");
    attnChart = new Chart(ctx3, {
        type: 'bar',
        data: {
            labels: Array.from({length: 24}, (_, i) => `t-${24-i}h`),
            datasets: [{
                label: 'Temporal Attention Score',
                data: Array.from({length: 24}, (_, i) => (Math.exp(i / 8) / 10).toFixed(3)),
                backgroundColor: '#38bdf8'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
            }
        }
    });
}

function loadXAI(model) {
    fetch("/research/api/explain/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: model })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            const shapList = data.shap_explanation.feature_contributions;
            shapChart.data.labels = shapList.map(s => s.feature.toUpperCase());
            shapChart.data.datasets[0].data = shapList.map(s => s.shap_value);
            shapChart.update();

            const ig = data.feature_attribution.group_breakdown;
            igPieChart.data.datasets[0].data = [ig.pollutants_percentage, ig.weather_percentage];
            igPieChart.update();

            const attnList = data.temporal_attention.attention_scores;
            attnChart.data.labels = attnList.map(a => a.time_lag);
            attnChart.data.datasets[0].data = attnList.map(a => a.attention_weight);
            attnChart.update();
        }
    });
}

function calculateHealthRoutes() {
    const src = document.getElementById("routeSource").value.split(",");
    const dst = document.getElementById("routeDest").value.split(",");

    const payload = {
        source_lat: parseFloat(src[0]),
        source_lng: parseFloat(src[1]),
        dest_lat: parseFloat(dst[0]),
        dest_lng: parseFloat(dst[1])
    };

    fetch("/research/api/routes/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            renderHealthRoutesGrid(data.routes);
        }
    });
}

function renderHealthRoutesGrid(routes) {
    const grid = document.getElementById("routesGrid");
    grid.innerHTML = "";

    Object.values(routes).forEach(r => {
        const card = document.createElement("div");
        card.className = "route-card";
        card.innerHTML = `
            <h4>${r.display_name}</h4>
            <div class="route-metric-item"><span class="route-metric-label">Distance:</span><span class="route-metric-val">${r.distance_km} km</span></div>
            <div class="route-metric-item"><span class="route-metric-label">Duration:</span><span class="route-metric-val">${r.duration_mins} mins</span></div>
            <div class="route-metric-item"><span class="route-metric-label">Avg AQI:</span><span class="route-metric-val">${r.average_aqi}</span></div>
            <div class="route-metric-item"><span class="route-metric-label">Inhalation Dosage:</span><span class="route-metric-val">${r.inhalation_dosage} units</span></div>
            <div class="route-metric-item"><span class="route-metric-label">Pollution Cost:</span><span class="route-metric-val">$${r.pollution_cost}</span></div>
            <div class="route-metric-item"><span class="route-metric-label">Health Score Index:</span><span class="badge ${r.health_score > 70 ? 'badge-success' : 'badge-warning'}">${r.health_score} / 100</span></div>
        `;
        grid.appendChild(card);
    });
}
