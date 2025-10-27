const state = {
  token: null,
  user: null,
  auctionId: localStorage.getItem('auctionId') || null,
  teamId: localStorage.getItem('teamId') || null,
  currentLot: null,
  bids: []
};

const elements = {
  loginForm: document.getElementById('login-form'),
  loginSection: document.getElementById('login-section'),
  adminSection: document.getElementById('admin-section'),
  ownerSection: document.getElementById('owner-section'),
  reportSection: document.getElementById('report-section'),
  authInfo: document.getElementById('auth-info'),
  auctionInfo: document.getElementById('auction-info'),
  auctionName: document.getElementById('auction-name'),
  playerImport: document.getElementById('player-import'),
  importPlayersButton: document.getElementById('import-players'),
  adminLot: document.getElementById('admin-lot'),
  adminSummary: document.getElementById('admin-summary'),
  setTeams: document.getElementById('set-teams'),
  createAuction: document.getElementById('create-auction'),
  queueLots: document.getElementById('queue-lots'),
  startAuction: document.getElementById('start-auction'),
  nextLot: document.getElementById('next-lot'),
  finalizeLot: document.getElementById('finalize-lot'),
  completeAuction: document.getElementById('complete-auction'),
  overrideComplete: document.getElementById('override-complete'),
  downloadReport: document.getElementById('download-report'),
  refreshSummary: document.getElementById('refresh-summary'),
  ownerBudget: document.getElementById('owner-budget'),
  ownerRoster: document.getElementById('owner-roster'),
  ownerLot: document.getElementById('owner-lot'),
  bidAmount: document.getElementById('bid-amount'),
  bidButtons: document.querySelectorAll('.bid-controls button[data-inc]'),
  submitBid: document.getElementById('submit-bid'),
  bidHistory: document.getElementById('bid-history'),
  loadOwnerState: document.getElementById('load-owner-state'),
  loadReport: document.getElementById('load-report'),
  reportOutput: document.getElementById('report-output')
};

let socket;

function updateAuthUI() {
  if (state.user) {
    elements.authInfo.textContent = `${state.user.username} (${state.user.role})`;
  } else {
    elements.authInfo.textContent = '';
  }
  if (state.auctionId) {
    elements.auctionInfo.textContent = `Auction ID: ${state.auctionId}`;
  } else {
    elements.auctionInfo.textContent = '';
  }
}

function showSection(section, show) {
  section.classList.toggle('hidden', !show);
}

function configureLayout() {
  const isAdmin = state.user?.role === 'ADMIN';
  const isOwner = state.user?.role === 'OWNER';
  showSection(elements.loginSection, !state.user);
  showSection(elements.adminSection, isAdmin);
  showSection(elements.ownerSection, isOwner);
  showSection(elements.reportSection, Boolean(state.user));
  updateAuthUI();
}

async function api(path, { method = 'GET', body } = {}) {
  if (!state.token) throw new Error('Missing auth');
  const res = await fetch(path, {
    method,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${state.token}`
    },
    body: body ? JSON.stringify(body) : undefined
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

function connectSocket() {
  if (!state.auctionId) return;
  if (socket) {
    socket.close();
  }
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${protocol}://${location.host}/?auctionId=${state.auctionId}`);
  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      handleSocketEvent(payload);
    } catch (err) {
      console.error('socket message error', err);
    }
  };
}

function handleSocketEvent({ event, payload }) {
  if (event === 'LOT_ACTIVATED') {
    state.currentLot = payload.lot;
    state.bids = [];
    renderLots();
  }
  if (event === 'BID_PLACED') {
    state.bids.unshift(payload.bid);
    renderBids();
  }
  if (event === 'LOT_SOLD') {
    state.currentLot = null;
    renderLots();
  }
  if (event === 'TEAM_UPDATED' || event === 'AUCTION_STATUS_CHANGED') {
    refreshSummary();
  }
}

async function login(username, password) {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  if (!res.ok) {
    throw new Error('Invalid credentials');
  }
  const data = await res.json();
  state.token = data.token;
  state.user = data.user;
  configureLayout();
  if (state.user.role === 'ADMIN') {
    await ensureAuctionSelected();
  }
  if (state.user.role === 'OWNER') {
    await ensureAuctionSelected();
    await initializeOwner();
  }
  connectSocket();
}

elements.loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(elements.loginForm);
  const username = formData.get('username');
  const password = formData.get('password');
  try {
    await login(username, password);
  } catch (err) {
    alert(err.message);
  }
});

async function ensureAuctionSelected() {
  if (state.auctionId) return;
  const input = prompt('Enter auction ID to use (or leave blank to create one).');
  if (input) {
    state.auctionId = input.trim();
    localStorage.setItem('auctionId', state.auctionId);
    updateAuthUI();
  }
}

elements.createAuction?.addEventListener('click', async () => {
  if (!elements.auctionName.value.trim()) {
    const name = prompt('Auction name');
    if (!name) return;
    elements.auctionName.value = name;
  }
  try {
    const response = await api('/api/auctions', {
      method: 'POST',
      body: {
        name: elements.auctionName.value || 'FC26 Auction',
        increment: 1000,
        antiSnipeThreshold: 5,
        antiSnipeExtension: 10
      }
    });
    state.auctionId = response.auction.id;
    localStorage.setItem('auctionId', state.auctionId);
    updateAuthUI();
    alert(`Auction created with id ${state.auctionId}`);
  } catch (err) {
    alert(err.message);
  }
});

elements.setTeams?.addEventListener('click', async () => {
  if (!state.auctionId) {
    alert('Create or select an auction first.');
    return;
  }
  const template = '[{"name":"Blue","ownerUsername":"owner1","budget":1000000}]';
  const text = prompt('Enter team definitions as JSON', template);
  if (!text) return;
  try {
    const teams = JSON.parse(text);
    await api(`/api/auctions/${state.auctionId}/teams`, { method: 'POST', body: { teams } });
    alert('Teams saved');
    refreshSummary();
  } catch (err) {
    alert(err.message);
  }
});

elements.importPlayersButton?.addEventListener('click', async () => {
  if (!state.auctionId) {
    alert('Create or select an auction first.');
    return;
  }
  const csv = elements.playerImport.value.trim();
  if (!csv) {
    alert('Paste a CSV payload first');
    return;
  }
  try {
    await api('/api/players/import', { method: 'POST', body: { data: csv } });
    alert('Players imported');
  } catch (err) {
    alert(err.message);
  }
});

elements.queueLots?.addEventListener('click', async () => {
  if (!state.auctionId) return alert('Select auction first');
  try {
    const playerData = await api('/api/players');
    if (!playerData.players.length) throw new Error('No players imported');
    await api(`/api/auctions/${state.auctionId}/lots/queue`, {
      method: 'POST',
      body: { playerIds: playerData.players.map((p) => p.id) }
    });
    alert('Players queued');
  } catch (err) {
    alert(err.message);
  }
});

elements.startAuction?.addEventListener('click', async () => {
  if (!state.auctionId) return alert('Select auction first');
  try {
    await api(`/api/auctions/${state.auctionId}/start`, { method: 'PATCH' });
    alert('Auction started');
  } catch (err) {
    alert(err.message);
  }
});

elements.nextLot?.addEventListener('click', async () => {
  if (!state.auctionId) return alert('Select auction first');
  try {
    const data = await api(`/api/auctions/${state.auctionId}/lots/next`, { method: 'PATCH' });
    state.currentLot = data.lot;
    state.bids = [];
    renderLots();
  } catch (err) {
    alert(err.message);
  }
});

elements.finalizeLot?.addEventListener('click', async () => {
  if (!state.auctionId || !state.currentLot) {
    alert('Activate a lot first');
    return;
  }
  try {
    await api(`/api/lots/${state.currentLot.id}/finalize`, {
      method: 'PATCH',
      body: { auctionId: state.auctionId }
    });
    state.currentLot = null;
    state.bids = [];
    renderLots();
    await refreshSummary();
  } catch (err) {
    alert(err.message);
  }
});

elements.completeAuction?.addEventListener('click', async () => {
  if (!state.auctionId) return alert('Select auction first');
  try {
    await api(`/api/auctions/${state.auctionId}/complete`, {
      method: 'PATCH',
      body: { override: false }
    });
    alert('Auction completed');
  } catch (err) {
    alert(err.message);
  }
});

elements.overrideComplete?.addEventListener('click', async () => {
  if (!state.auctionId) return alert('Select auction first');
  try {
    await api(`/api/auctions/${state.auctionId}/complete`, {
      method: 'PATCH',
      body: { override: true }
    });
    alert('Auction completion overridden');
  } catch (err) {
    alert(err.message);
  }
});

elements.downloadReport?.addEventListener('click', async () => {
  if (!state.auctionId) return alert('Select auction first');
  try {
    const data = await api(`/api/auctions/${state.auctionId}/export.csv`);
    const blob = new Blob([data.csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `auction-${state.auctionId}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(err.message);
  }
});

elements.refreshSummary?.addEventListener('click', () => {
  refreshSummary();
});

async function refreshSummary() {
  if (!state.auctionId || !state.token) return;
  try {
    const data = await api(`/api/auctions/${state.auctionId}/reports/teams-summary`);
    elements.adminSummary.textContent = JSON.stringify(data.summary, null, 2);
    if (state.user?.role === 'OWNER') {
      const team = data.summary.find((entry) => entry.team.ownerUserId === state.user.id || entry.team.id === state.teamId);
      if (team) {
        state.teamId = team.team.id;
        localStorage.setItem('teamId', state.teamId);
        elements.ownerBudget.textContent = `${team.remainingBudget.toLocaleString()} credits`;
        elements.ownerRoster.innerHTML = team.players
          .map((p) => `<li>${p.player?.name || 'Unknown'} - ${p.soldPrice?.toLocaleString() || ''}</li>`)
          .join('');
      }
    }
  } catch (err) {
    console.warn('summary error', err);
  }
}

async function loadActiveLot() {
  if (!state.auctionId) return;
  try {
    const data = await api(`/api/auctions/${state.auctionId}/lots/active`);
    state.currentLot = data.lot;
    state.bids = data.bids || [];
    renderLots();
  } catch (err) {
    console.warn('lot load error', err);
  }
}

async function initializeOwner() {
  if (!state.auctionId) return;
  try {
    const teamData = await api(`/api/auctions/${state.auctionId}/teams`);
    const team = teamData.teams.find((t) => t.ownerUserId === state.user.id);
    if (team) {
      state.teamId = team.id;
      localStorage.setItem('teamId', state.teamId);
    }
  } catch (err) {
    console.warn('team lookup error', err);
  }
  await refreshSummary();
  await loadActiveLot();
}

elements.bidButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const inc = Number(btn.dataset.inc);
    elements.bidAmount.value = (Number(elements.bidAmount.value || state.currentLot?.reserve || 0) + inc).toString();
  });
});

elements.submitBid?.addEventListener('click', async () => {
  if (!state.auctionId || !state.teamId || !state.currentLot) {
    alert('Missing auction, team, or lot');
    return;
  }
  const amount = Number(elements.bidAmount.value);
  if (!amount) {
    alert('Enter a bid amount');
    return;
  }
  try {
    const response = await api(`/api/lots/${state.currentLot.id}/bids`, {
      method: 'POST',
      body: { amount, auctionId: state.auctionId, teamId: state.teamId }
    });
    state.bids.unshift(response.bid);
    renderBids();
  } catch (err) {
    alert(err.message);
  }
});

elements.loadOwnerState?.addEventListener('click', async () => {
  await refreshSummary();
  await loadActiveLot();
});

elements.loadReport?.addEventListener('click', async () => {
  if (!state.auctionId || !state.token) return;
  try {
    const [highest, summary] = await Promise.all([
      api(`/api/auctions/${state.auctionId}/reports/highest-sale`),
      api(`/api/auctions/${state.auctionId}/reports/teams-summary`)
    ]);
    elements.reportOutput.textContent = JSON.stringify({ highest: highest.report, summary: summary.summary }, null, 2);
  } catch (err) {
    alert(err.message);
  }
});

function renderLots() {
  elements.adminLot.textContent = state.currentLot ? JSON.stringify(state.currentLot, null, 2) : 'No active lot';
  elements.ownerLot.textContent = state.currentLot ? JSON.stringify(state.currentLot, null, 2) : 'No active lot';
  renderBids();
}

function renderBids() {
  elements.bidHistory.innerHTML = state.bids
    .map((bid) => `<li>${new Date(bid.placedAt).toLocaleTimeString()} - ${bid.amount.toLocaleString()} by ${bid.teamId}</li>`)
    .join('');
}

configureLayout();
updateAuthUI();
if (state.auctionId) {
  connectSocket();
}
