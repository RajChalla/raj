import { test } from 'node:test';
import assert from 'node:assert';
import { computeBasePrice } from '../src/utils/basePrice.js';

const tiers = [
  { min: 90, max: 99, price: 300000 },
  { min: 87, max: 89, price: 200000 },
  { min: 84, max: 86, price: 120000 },
  { min: 81, max: 83, price: 70000 },
  { min: 78, max: 80, price: 40000 },
  { min: 75, max: 77, price: 20000 },
  { min: 0, max: 74, price: 10000 }
];

test('base price uses matching tier', () => {
  assert.strictEqual(computeBasePrice(92, tiers), 300000);
  assert.strictEqual(computeBasePrice(88, tiers), 200000);
  assert.strictEqual(computeBasePrice(82, tiers), 70000);
  assert.strictEqual(computeBasePrice(50, tiers), 10000);
});
