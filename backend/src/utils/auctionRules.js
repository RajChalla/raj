export function canCompleteAuction({ teams, rosterMap }) {
  const violations = [];
  for (const team of teams) {
    const rosterCount = rosterMap.get(team.id) || 0;
    if (rosterCount < 19) {
      violations.push({ teamId: team.id, teamName: team.name, rosterCount });
    }
  }
  return {
    ok: violations.length === 0,
    violations
  };
}
