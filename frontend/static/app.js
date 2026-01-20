// State
let currentSource = null;

// DOM Elements
const chatHistory = document.getElementById('chat-history');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const activeDataList = document.getElementById('active-data-list');

// --- Navigation ---
function switchTab(tabName) {
    // Hide all views
    document.querySelectorAll('.view-section').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

    // Show selected
    document.getElementById(`view-${tabName}`).style.display = 'block';

    // Highlight nav button (simple logic, assuming order matches)
    const btnIndex = ['analysis', 'sources', 'settings'].indexOf(tabName);
    if (btnIndex >= 0) {
        document.querySelectorAll('.nav-btn')[btnIndex].classList.add('active');
    }
}

// --- Chat Functions ---
function appendMessage(role, content) {
    const isUser = role === 'user';
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;

    const avatar = isUser ? '👤' : '🤖';

    let htmlContent = content;
    if (!isUser) {
        // Parse basic markdown if AI
        htmlContent = marked.parse(content);
    }

    msgDiv.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="message-content">${htmlContent}</div>
    `;

    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function appendPlot(filename) {
    const id = `plot-${Date.now()}`;
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai-message';
    msgDiv.innerHTML = `
        <div class="avatar">📊</div>
        <div class="message-content">
            <p>Here is the visualization:</p>
            <div class="plot-container">
                <img src="/api/plot/${filename}" alt="Analysis Plot" loading="lazy" />
            </div>
        </div>
    `;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function clearChat() {
    chatHistory.innerHTML = '';
    // Add welcome back
    appendMessage('ai', 'Chat cleared. Ready for new questions.');
}

async function handleChatSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('user-input');
    const query = input.value.trim();
    if (!query) return;

    if (!currentSource) {
        appendMessage('user', query);
        setTimeout(() => appendMessage('ai', '⚠️ Please connect a data source first (Go to "Data Sources" tab).'), 500);
        input.value = '';
        return;
    }

    // Add user message
    appendMessage('user', query);
    input.value = '';

    // Show loading state
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message ai-message';
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="message-content">Thinking... <i class="fa-solid fa-spinner fa-spin"></i></div>
    `;
    chatHistory.appendChild(loadingDiv);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        // Remove loading
        document.getElementById(loadingId).remove();

        if (data.success) {
            appendMessage('ai', data.summary || "Analysis complete.");

            if (data.plot_filename) {
                appendPlot(data.plot_filename);
            }
        } else {
            appendMessage('ai', `❌ Error: ${data.msg || data.error}`);
        }

    } catch (err) {
        document.getElementById(loadingId).remove();
        appendMessage('ai', `❌ Network Error: ${err.message}`);
    }
}

// --- File Upload ---
fileInput.addEventListener('change', (e) => handleFileUpload(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent-primary)';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
    if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

async function handleFileUpload(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const isPdf = file.name.toLowerCase().endsWith('.pdf');
    const endpoint = isPdf ? '/api/upload_pdf' : '/api/upload_csv';

    uploadStatus.innerHTML = '<span style="color:var(--text-muted)">Uploading... <i class="fa-solid fa-spinner fa-spin"></i></span>';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            uploadStatus.innerHTML = `<span style="color:var(--success)">✅ ${file.name} Uploaded!</span>`;
            currentSource = { name: file.name, type: isPdf ? 'pdf' : 'csv' };
            updateActiveList();

            // Auto switch to chat
            setTimeout(() => {
                switchTab('analysis');
                appendMessage('ai', `I've loaded **${file.name}**. What would you like to know?`);
            }, 1000);
        } else {
            uploadStatus.innerHTML = `<span style="color:var(--error)">❌ ${data.msg}</span>`;
        }
    } catch (err) {
        uploadStatus.innerHTML = '<span style="color:var(--error)">❌ Upload failed</span>';
        console.error(err);
    }
}

// --- Database Connection ---
const dbModal = document.getElementById('db-modal');

function showDbModal() { dbModal.classList.add('open'); }
function closeDbModal() { dbModal.classList.remove('open'); }

function updateDbForm() {
    const dbType = document.getElementById('db-type').value;
    const mysqlFields = document.getElementById('mysql-fields');
    const postgresFields = document.getElementById('postgres-fields');

    if (dbType === 'mysql') {
        mysqlFields.style.display = 'block';
        postgresFields.style.display = 'none';
        // Make MySQL fields required
        document.getElementById('mysql-host').required = true;
        document.getElementById('mysql-user').required = true;
        document.getElementById('mysql-password').required = true;
        document.getElementById('mysql-database').required = true;
        document.getElementById('mysql-table').required = true;
        // Make PostgreSQL fields optional
        document.getElementById('postgres-url').required = false;
        document.getElementById('postgres-table').required = false;
    } else if (dbType === 'postgresql') {
        mysqlFields.style.display = 'none';
        postgresFields.style.display = 'block';
        // Make PostgreSQL fields required
        document.getElementById('postgres-url').required = true;
        document.getElementById('postgres-table').required = true;
        // Make MySQL fields optional
        document.getElementById('mysql-host').required = false;
        document.getElementById('mysql-user').required = false;
        document.getElementById('mysql-password').required = false;
        document.getElementById('mysql-database').required = false;
        document.getElementById('mysql-table').required = false;
    } else {
        mysqlFields.style.display = 'none';
        postgresFields.style.display = 'none';
    }
}

async function handleDbConnect(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    const dbType = data.db_type;

    // Loading State
    const submitBtn = e.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fa-solid fa-circle-notch spinner"></i> Connecting...';

    try {
        let requestBody;

        if (dbType === 'mysql') {
            requestBody = {
                db_type: 'mysql',
                host: data.host,
                port: data.port,
                user: data.user,
                password: data.password,
                database: data.database,
                table: data.table
            };
        } else if (dbType === 'postgresql') {
            requestBody = {
                db_type: 'postgresql',
                postgres_url: data.postgres_url,
                table: data.table
            };
        } else {
            alert('Please select a database type');
            return;
        }

        const response = await fetch('/api/load_db', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        const result = await response.json();

        if (result.success) {
            closeDbModal();
            currentSource = { name: data.table, type: dbType };
            updateActiveList();
            switchTab('analysis');
            appendMessage('ai', `Connected to ${dbType.toUpperCase()} database table **${data.table}**. Ready!`);
        } else {
            alert('Connection Failed: ' + result.msg);
        }
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}

// --- UI Utilities ---
function updateActiveList() {
    if (!currentSource) return;

    const badgeClass = `badge-${currentSource.type}`;
    const icon = currentSource.type === 'pdf' ? 'fa-file-pdf' : (currentSource.type === 'csv' ? 'fa-file-csv' : 'fa-database');

    activeDataList.innerHTML = `
        <div class="source-item">
            <i class="fa-solid ${icon}"></i>
            <span>${currentSource.name}</span>
            <span class="source-type-badge ${badgeClass}">${currentSource.type.toUpperCase()}</span>
        </div>
    `;
}

// Initial Setup
switchTab('analysis'); // Start on chat view logic
switchTab('sources'); // But show sources first to prompt upload
