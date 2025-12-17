function showCsvUpload() {
    document.getElementById('csvForm').style.display = 'block';
    document.getElementById('dbForm').style.display = 'none';
    document.getElementById('output-container').style.display = 'none';
    // Update active button state
    document.querySelectorAll('.method-btn').forEach(btn => btn.classList.remove('active'));
    event.target.closest('.method-btn')?.classList.add('active');
}
function showDbForm() {
    document.getElementById('dbForm').style.display = 'block';
    document.getElementById('csvForm').style.display = 'none';
    document.getElementById('output-container').style.display = 'none';
    // Update active button state
    document.querySelectorAll('.method-btn').forEach(btn => btn.classList.remove('active'));
    event.target.closest('.method-btn')?.classList.add('active');
}

document.getElementById('csvForm').onsubmit = async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('csvfile');
    if (!fileInput.files[0]) return;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    setStatus('Uploading CSV...', 'info');
    const resp = await fetch('/api/upload_csv', {
        method: 'POST',
        body: formData
    });
    const data = await resp.json();
    if (data.success) {
        setStatus('CSV Loaded!', 'success');
        document.getElementById('queryForm').style.display = 'block';
        document.getElementById('output-container').style.display = 'none';
    } else {
        setStatus('Error: ' + data.msg, 'error');
    }
};

document.getElementById('dbForm').onsubmit = async (e) => {
    e.preventDefault();
    const host = document.getElementById('dbhost').value;
    const port = document.getElementById('dbport').value;
    const user = document.getElementById('dbuser').value;
    const password = document.getElementById('dbpass').value;
    const database = document.getElementById('dbname').value;
    setStatus('Connecting to DB...', 'info');
    const resp = await fetch('/api/load_db', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ host, port, user, password, database })
    });
    const data = await resp.json();
    if (data.success) {
        setStatus('Database Loaded!', 'success');
        document.getElementById('queryForm').style.display = 'block';
        document.getElementById('output-container').style.display = 'none';
    } else {
        setStatus('Error: ' + data.msg, 'error');
    }
};

document.getElementById('queryForm').onsubmit = async (e) => {
    e.preventDefault();
    const query = document.getElementById('query').value;
    setStatus('Analyzing...', 'info');
    document.getElementById('output-container').style.display = 'none';
    const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query })
    });
    const data = await resp.json();
    if (data.success) {
        showAnalysisResult(data);
        setStatus('Analysis Complete!', 'success');
    } else {
        setStatus('Error: ' + data.msg, 'error');
        document.getElementById('output-container').style.display = 'none';
    }
};

function setStatus(msg, type) {
    const el = document.getElementById('status');
    el.innerText = msg;
    el.className = type;
}

function showAnalysisResult(data) {
    document.getElementById('output-container').style.display = 'block';
    let html = '';
    
    // Show output text (the printed console output)
    if (data.output && data.output.trim()) {
        html += `<div class="output-section"><h3>Output:</h3><pre class="output-text">${escapeHtml(data.output)}</pre></div>`;
    }
    
    // Show result value if available
    if (data.result !== null && data.result !== undefined) {
        html += `<div class="result-section"><h3>Result:</h3><div class="result-value">${escapeHtml(String(data.result))}</div></div>`;
    }
    
    // Show plot if available
    if (data.plot_filename) {
        html += `<div class="plot-section"><h3>Visualization:</h3><img id="plotImg" class="plot-image" src="/api/plot/${data.plot_filename}" alt="Analysis Plot" style="display: block;" /></div>`;
    }
    
    document.getElementById('analysisResult').innerHTML = html || '<div class="no-result">No output generated.</div>';
    
    // Ensure plot image is properly sized after load
    const plotImg = document.getElementById('plotImg');
    if (plotImg) {
        plotImg.onload = function() {
            this.style.maxWidth = '100%';
            this.style.width = 'auto';
            this.style.height = 'auto';
        };
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

