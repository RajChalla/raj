export function validateBid({ amount, basePrice, currentHigh, remainingBudget, increment }) {
  if (amount < basePrice) {
    return 'Bid must meet or exceed reserve price';
  }
  if (amount > remainingBudget) {
    return 'Bid exceeds remaining budget';
  }
  if (currentHigh !== null && amount <= currentHigh) {
    return 'Bid must exceed current high bid';
  }
  if (currentHigh !== null && (amount - currentHigh) % increment !== 0) {
    return `Bid must increase by increments of ${increment}`;
  }
  if (currentHigh === null && (amount - basePrice) % increment !== 0) {
    return `First bid must align with increment ${increment}`;
  }
  return null;
}
