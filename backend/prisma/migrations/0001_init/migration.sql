-- CreateEnum
CREATE TYPE "Role" AS ENUM ('ADMIN', 'OWNER', 'VIEWER');

CREATE TYPE "AuctionStatus" AS ENUM ('DRAFT', 'RUNNING', 'PAUSED', 'COMPLETED');
CREATE TYPE "LotStatus" AS ENUM ('QUEUED', 'ACTIVE', 'SOLD', 'UNSOLD');

-- CreateTable
CREATE TABLE "User" (
    "id" TEXT PRIMARY KEY,
    "email" TEXT NOT NULL UNIQUE,
    "passwordHash" TEXT NOT NULL,
    "role" "Role" NOT NULL,
    "displayName" TEXT NOT NULL,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "Auction" (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "status" "AuctionStatus" NOT NULL DEFAULT 'DRAFT',
    "startTime" TIMESTAMP,
    "endTime" TIMESTAMP,
    "antiSnipeWindow" INTEGER NOT NULL DEFAULT 5,
    "antiSnipeExtend" INTEGER NOT NULL DEFAULT 10,
    "bidIncrement" INTEGER NOT NULL DEFAULT 1000,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "settingsSnapshot" JSONB
);

CREATE TABLE "Team" (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "auctionId" TEXT NOT NULL REFERENCES "Auction"("id") ON DELETE CASCADE,
    "ownerId" TEXT REFERENCES "User"("id") ON DELETE SET NULL,
    "initialBudget" INTEGER NOT NULL,
    "remainingBudget" INTEGER NOT NULL,
    "rosterCount" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "Player" (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "position" TEXT NOT NULL,
    "club" TEXT NOT NULL,
    "nation" TEXT NOT NULL,
    "rating" INTEGER NOT NULL,
    "rarity" TEXT NOT NULL,
    "gender" TEXT NOT NULL,
    "basePrice" INTEGER NOT NULL,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "Lot" (
    "id" TEXT PRIMARY KEY,
    "auctionId" TEXT NOT NULL REFERENCES "Auction"("id") ON DELETE CASCADE,
    "playerId" TEXT NOT NULL REFERENCES "Player"("id") ON DELETE CASCADE,
    "status" "LotStatus" NOT NULL DEFAULT 'QUEUED',
    "reservePrice" INTEGER NOT NULL,
    "currentBidId" TEXT,
    "startsAt" TIMESTAMP,
    "endsAt" TIMESTAMP,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "Bid" (
    "id" TEXT PRIMARY KEY,
    "lotId" TEXT NOT NULL REFERENCES "Lot"("id") ON DELETE CASCADE,
    "teamId" TEXT NOT NULL REFERENCES "Team"("id") ON DELETE CASCADE,
    "amount" INTEGER NOT NULL,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "Sale" (
    "id" TEXT PRIMARY KEY,
    "lotId" TEXT NOT NULL UNIQUE REFERENCES "Lot"("id") ON DELETE CASCADE,
    "playerId" TEXT NOT NULL REFERENCES "Player"("id") ON DELETE CASCADE,
    "auctionId" TEXT NOT NULL REFERENCES "Auction"("id") ON DELETE CASCADE,
    "teamId" TEXT REFERENCES "Team"("id") ON DELETE SET NULL,
    "basePrice" INTEGER NOT NULL,
    "soldPrice" INTEGER,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "AuditLog" (
    "id" TEXT PRIMARY KEY,
    "auctionId" TEXT REFERENCES "Auction"("id") ON DELETE SET NULL,
    "teamId" TEXT REFERENCES "Team"("id") ON DELETE SET NULL,
    "userId" TEXT REFERENCES "User"("id") ON DELETE SET NULL,
    "action" TEXT NOT NULL,
    "detail" JSONB NOT NULL,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE "Setting" (
    "id" TEXT PRIMARY KEY,
    "key" TEXT NOT NULL UNIQUE,
    "value" JSONB NOT NULL,
    "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE "Lot" ADD CONSTRAINT "Lot_currentBidId_fkey" FOREIGN KEY ("currentBidId") REFERENCES "Bid"("id") ON DELETE SET NULL;

CREATE INDEX "Bid_lotId_idx" ON "Bid"("lotId");
CREATE INDEX "Bid_teamId_idx" ON "Bid"("teamId");
