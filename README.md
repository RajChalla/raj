# FC26 Men's Gold Rare Auction Platform

This repository hosts a full-stack real-time auction platform for FC26 Men's Gold Rare player cards. It includes:

- **Node.js + Express + Prisma** backend with Socket.IO and PostgreSQL
- **React + Vite** frontend for admin and team owners
- **Docker Compose** for local development and deployment
- Comprehensive business logic covering auction rules, roster minimums, anti-snipe extensions, and reporting

## Getting Started

### Prerequisites

- Docker & Docker Compose

### Environment Setup

Copy the sample environment file:

```bash
cp .env.example .env
```

Update values as needed (especially `JWT_SECRET`).

### Launching the stack

```bash
docker-compose up --build
```

The services will be available at:

- Backend API: `http://localhost:4000`
- Frontend: `http://localhost:5173`
- PostgreSQL: `localhost:5432`

After the containers start, apply the database schema:

```bash
docker-compose exec api npx prisma migrate deploy
```

### Seed Data

Run the seed script inside the API container to load sample Gold Rare players and demo accounts:

```bash
docker-compose run --rm api npm install
docker-compose run --rm api npm run seed
```

(Alternatively, import your own player CSV/JSON through the admin UI.)

### Demo Accounts

| Role  | Email                | Password    |
|-------|----------------------|-------------|
| Admin | admin@example.com    | adminpass   |
| Owner | owner1@example.com   | owner1pass  |
| Owner | owner2@example.com   | owner2pass  |

### Testing

From the `backend` directory run:

```bash
npm install
npm test
```

Tests cover base price tiers, bid validation rules, roster completion guard, and an end-to-end bidding flow.

### Development Scripts

- `npm run dev` in `backend`: start API with hot reload
- `npm run dev` in `frontend`: start Vite dev server

## API Overview

Key endpoints include:

- `POST /api/auth/login` – authenticate via email/password
- `POST /api/players/import` – admin import of Gold Rare player catalog
- `POST /api/auctions` – create auctions with per-team budgets
- `POST /api/lots/:lotId/bids` – place real-time bids with server-side validation
- `GET /api/auctions/:id/reports/teams-summary` – post-auction reporting with CSV export

Refer to source code for the full list of routes and payloads.

## License

MIT
