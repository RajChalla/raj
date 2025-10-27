import { test } from 'node:test';
import assert from 'node:assert';
import { validateBid } from '../src/utils/bidRules.js';

test('rejects bids below reserve', () => {
  assert.strictEqual(
    validateBid({ amount: 5000, basePrice: 10000, currentHigh: null, remainingBudget: 20000, increment: 1000 }),
    'Bid must meet or exceed reserve price'
  );
});

test('rejects bids exceeding budget', () => {
  assert.strictEqual(
    validateBid({ amount: 21000, basePrice: 10000, currentHigh: 15000, remainingBudget: 20000, increment: 1000 }),
    'Bid exceeds remaining budget'
  );
});

test('rejects bids not respecting increment', () => {
  assert.strictEqual(
    validateBid({ amount: 16050, basePrice: 10000, currentHigh: 15000, remainingBudget: 50000, increment: 1000 }),
    'Bid must increase by increments of 1000'
  );
});

test('accepts valid bid', () => {
  assert.strictEqual(
    validateBid({ amount: 16000, basePrice: 10000, currentHigh: 15000, remainingBudget: 50000, increment: 1000 }),
    null
  );
});
