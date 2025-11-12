import { test } from 'node:test';
import assert from 'node:assert';
import { canCompleteAuction } from '../src/utils/auctionRules.js';

test('requires teams to meet roster minimum', () => {
  const teams = [
    { id: 'team1', name: 'Alpha' },
    { id: 'team2', name: 'Beta' }
  ];
  const rosterMap = new Map([
    ['team1', 19],
    ['team2', 18]
  ]);
  const result = canCompleteAuction({ teams, rosterMap });
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.violations.length, 1);
  assert.strictEqual(result.violations[0].teamId, 'team2');
});
