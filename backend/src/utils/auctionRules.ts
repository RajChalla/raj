import { Team } from '@prisma/client';

export function teamsBelowMinimum(teams: Pick<Team, 'id' | 'rosterCount'>[], minimum = 19) {
  return teams.filter((team) => team.rosterCount < minimum);
}
