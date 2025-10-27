const players = [
  {
    id: 1,
    name: "Erling Haaland",
    club: "Manchester City",
    nation: "Norway",
    position: "FWD",
    rating: 94,
    price: 24.5,
  },
  {
    id: 2,
    name: "Kylian Mbappé",
    club: "Paris SG",
    nation: "France",
    position: "FWD",
    rating: 95,
    price: 27.0,
  },
  {
    id: 3,
    name: "Jude Bellingham",
    club: "Real Madrid",
    nation: "England",
    position: "MID",
    rating: 93,
    price: 21.0,
  },
  {
    id: 4,
    name: "Kevin De Bruyne",
    club: "Manchester City",
    nation: "Belgium",
    position: "MID",
    rating: 92,
    price: 19.5,
  },
  {
    id: 5,
    name: "Vinícius Jr.",
    club: "Real Madrid",
    nation: "Brazil",
    position: "FWD",
    rating: 93,
    price: 22.0,
  },
  {
    id: 6,
    name: "Lionel Messi",
    club: "Inter Miami",
    nation: "Argentina",
    position: "FWD",
    rating: 92,
    price: 18.0,
  },
  {
    id: 7,
    name: "Rodri",
    club: "Manchester City",
    nation: "Spain",
    position: "MID",
    rating: 91,
    price: 17.0,
  },
  {
    id: 8,
    name: "Declan Rice",
    club: "Arsenal",
    nation: "England",
    position: "MID",
    rating: 90,
    price: 15.5,
  },
  {
    id: 9,
    name: "Trent Alexander-Arnold",
    club: "Liverpool",
    nation: "England",
    position: "DEF",
    rating: 89,
    price: 13.0,
  },
  {
    id: 10,
    name: "Ronald Araújo",
    club: "Barcelona",
    nation: "Uruguay",
    position: "DEF",
    rating: 90,
    price: 14.0,
  },
  {
    id: 11,
    name: "Thibaut Courtois",
    club: "Real Madrid",
    nation: "Belgium",
    position: "GK",
    rating: 91,
    price: 16.5,
  },
  {
    id: 12,
    name: "Gianluigi Donnarumma",
    club: "Paris SG",
    nation: "Italy",
    position: "GK",
    rating: 90,
    price: 15.0,
  },
  {
    id: 13,
    name: "Josko Gvardiol",
    club: "Manchester City",
    nation: "Croatia",
    position: "DEF",
    rating: 89,
    price: 12.5,
  },
  {
    id: 14,
    name: "Pedri",
    club: "Barcelona",
    nation: "Spain",
    position: "MID",
    rating: 90,
    price: 15.0,
  },
  {
    id: 15,
    name: "Jamal Musiala",
    club: "Bayern Munich",
    nation: "Germany",
    position: "MID",
    rating: 91,
    price: 18.5,
  },
];

const MAX_BUDGET = 100;
const draft = new Map();

const selectors = {
  players: document.querySelector("#players"),
  draftList: document.querySelector("#draftList"),
  remainingBudget: document.querySelector("#remainingBudget"),
  squadCount: document.querySelector("#squadCount"),
  averageRating: document.querySelector("#averageRating"),
  searchInput: document.querySelector("#searchInput"),
  positionFilter: document.querySelector("#positionFilter"),
  ratingRange: document.querySelector("#ratingRange"),
  ratingValue: document.querySelector("#ratingValue"),
  sortSelect: document.querySelector("#sortSelect"),
  clearTeam: document.querySelector("#clearTeam"),
  toggleDraftBoard: document.querySelector("#toggleDraftBoard"),
  draftBoard: document.querySelector("#team"),
};

function formatMillions(value) {
  return `$${value.toFixed(1)}M`;
}

function calculateRemainingBudget() {
  const spent = Array.from(draft.values()).reduce((sum, player) => sum + player.price, 0);
  return MAX_BUDGET - spent;
}

function calculateAverageRating() {
  if (draft.size === 0) return null;
  const total = Array.from(draft.values()).reduce((sum, player) => sum + player.rating, 0);
  return total / draft.size;
}

function updateDashboard() {
  const remaining = calculateRemainingBudget();
  selectors.remainingBudget.textContent = formatMillions(remaining);
  selectors.squadCount.textContent = draft.size;
  const avg = calculateAverageRating();
  selectors.averageRating.textContent = avg ? avg.toFixed(1) : "—";

  const playerCards = selectors.players.querySelectorAll(".player-card");
  playerCards.forEach((card) => {
    const price = parseFloat(card.dataset.price);
    const id = parseInt(card.dataset.id, 10);
    const button = card.querySelector(".add-button");

    if (draft.has(id)) {
      card.classList.add("disabled");
      button.classList.add("added");
      button.textContent = "In squad";
    } else {
      card.classList.remove("disabled");
      button.classList.remove("added");
      button.textContent = "Add to squad";
    }

    if (!draft.has(id) && remaining < price) {
      card.classList.add("disabled");
    }
  });
}

function renderPlayers(list) {
  selectors.players.innerHTML = "";
  const template = document.querySelector("#playerCardTemplate");

  list.forEach((player) => {
    const node = template.content.cloneNode(true);
    const card = node.querySelector(".player-card");
    card.dataset.id = player.id;
    card.dataset.price = player.price;

    node.querySelector(".player-rating").textContent = player.rating;
    node.querySelector(".player-name").textContent = player.name;
    node.querySelector(".player-club").textContent = player.club;
    node.querySelector(".player-price").textContent = formatMillions(player.price);
    node.querySelector(".badge.position").textContent = player.position;
    node.querySelector(".badge.nation").textContent = player.nation;

    node.querySelector(".add-button").addEventListener("click", () => addPlayer(player.id));

    selectors.players.appendChild(node);
  });

  updateDashboard();
}

function renderDraft() {
  selectors.draftList.innerHTML = "";
  const template = document.querySelector("#draftItemTemplate");

  if (draft.size === 0) {
    selectors.draftList.innerHTML = `<p class="empty">No players drafted yet. Use the board to add your first star.</p>`;
    return;
  }

  draft.forEach((player) => {
    const node = template.content.cloneNode(true);
    node.querySelector(".draft-name").textContent = player.name;
    node.querySelector(".draft-position").textContent = player.position;
    node.querySelector(".draft-rating").textContent = `⭐ ${player.rating}`;
    node.querySelector(".draft-price").textContent = formatMillions(player.price);
    node.querySelector(".remove-button").addEventListener("click", () => removePlayer(player.id));
    selectors.draftList.appendChild(node);
  });
}

function addPlayer(id) {
  if (draft.has(id)) return;
  const player = players.find((p) => p.id === id);
  if (!player) return;

  const remaining = calculateRemainingBudget();
  if (player.price > remaining) {
    highlightBudgetWarning();
    return;
  }

  draft.set(id, player);
  renderDraft();
  updateDashboard();
}

function removePlayer(id) {
  draft.delete(id);
  renderDraft();
  updateDashboard();
}

function highlightBudgetWarning() {
  selectors.remainingBudget.classList.add("shake");
  setTimeout(() => selectors.remainingBudget.classList.remove("shake"), 600);
}

function filterPlayers() {
  const searchTerm = selectors.searchInput.value.toLowerCase();
  const position = selectors.positionFilter.value;
  const minimumRating = parseInt(selectors.ratingRange.value, 10);
  const sort = selectors.sortSelect.value;

  const filtered = players
    .filter((player) => {
      const matchesSearch =
        player.name.toLowerCase().includes(searchTerm) ||
        player.club.toLowerCase().includes(searchTerm);
      const matchesPosition = position === "all" || player.position === position;
      const matchesRating = player.rating >= minimumRating;
      return matchesSearch && matchesPosition && matchesRating;
    })
    .sort((a, b) => {
      switch (sort) {
        case "rating-asc":
          return a.rating - b.rating;
        case "price-asc":
          return a.price - b.price;
        case "price-desc":
          return b.price - a.price;
        case "rating-desc":
        default:
          return b.rating - a.rating;
      }
    });

  renderPlayers(filtered);
}

function resetSquad() {
  draft.clear();
  renderDraft();
  updateDashboard();
}

function toggleDraftBoard() {
  const isOpen = selectors.draftBoard.classList.toggle("open");
  selectors.toggleDraftBoard.textContent = isOpen ? "Hide Draft Board" : "Open Draft Board";
  selectors.toggleDraftBoard.setAttribute("aria-expanded", String(isOpen));
}

function init() {
  selectors.draftBoard.classList.add("open");
  selectors.toggleDraftBoard.textContent = "Hide Draft Board";
  selectors.toggleDraftBoard.setAttribute("aria-expanded", "true");

  selectors.ratingRange.addEventListener("input", () => {
    selectors.ratingValue.textContent = selectors.ratingRange.value;
    filterPlayers();
  });

  selectors.searchInput.addEventListener("input", filterPlayers);
  selectors.positionFilter.addEventListener("change", filterPlayers);
  selectors.sortSelect.addEventListener("change", filterPlayers);
  selectors.clearTeam.addEventListener("click", resetSquad);
  selectors.toggleDraftBoard.addEventListener("click", toggleDraftBoard);

  renderPlayers(players.sort((a, b) => b.rating - a.rating));
  renderDraft();
  updateDashboard();
}

document.addEventListener("DOMContentLoaded", init);
