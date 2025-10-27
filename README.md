# FC26 Auction Platform

This repository contains a self-contained implementation of the FC26 Gold Rare auction platform. It ships with a lightweight Node.js backend, a static frontend, offline-friendly tests, and Docker orchestration. No external package registry access is required.

## Features

- Plain-text JSON datastore persisted to `backend/data/store.json`; no external database is required.
- Preloaded text-file roster of the top 100 FC26 men’s Gold Rare players in `backend/data/players.json` for immediate auctioning.
- Player import that filters for FC26 **men’s Gold Rare** cards and computes base prices from configurable tiers.
- Auction lifecycle with queues, single active lot enforcement, bid validation, anti-snipe extensions, reserve handling, unsold marking, and manual overrides logged through the audit log.
- Per-auction budgets, roster minimum enforcement (19 players), and override auditing.
- Real-time updates delivered via a minimal WebSocket implementation with `LOT_ACTIVATED`, `BID_PLACED`, `LOT_SOLD`, and `AUCTION_STATUS_CHANGED` events.
- Admin/owner/report dashboards built with vanilla JavaScript that interact with the API and socket layer.
- CSV export and JSON reports for highest sale and team summaries.
- Unit and integration tests executed with the built-in `node:test` runner.
- Docker Compose stack (`api`, `web`) requiring no package downloads.

## Getting Started

### Prerequisites
- Node.js 18+

### Local Development

```bash
# start the API and static frontend
cd backend
npm start
```

Open http://localhost:4000/ to access the web console.

Demo credentials are seeded automatically:
- Admin: `admin` / `admin123`
- Owner 1: `owner1` / `owner123`
- Owner 2: `owner2` / `owner123`
- Viewer: `viewer` / `viewer123`

### Tests

```bash
cd backend
npm test
```

### Docker

```
docker compose up
```

The compose stack builds the API with no dependency downloads and serves the static frontend.

### Environment Variables

See `.env.example` for optional overrides. The server defaults to `PORT=4000` and uses `APP_SECRET` for signing JWT-like tokens.

### Project Layout

```
backend/
  data/             # persisted datastore
    players.json    # top 100 FC26 Gold Rare men, used to seed the store
  src/              # backend source code
  tests/            # node:test suites
web/                # static frontend assets
```

