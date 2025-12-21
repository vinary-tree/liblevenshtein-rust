// Phonetic Spellcheck WASM Demo
// This module handles the browser UI and communicates with the WASM module.

import init, {
    initSpellchecker,
    query,
    get_stats,
    clear_cache,
    is_initialized
} from '../pkg/phonetic_spellcheck_wasm.js';

// DOM elements
const loadingEl = document.getElementById('loading');
const loadingDetailEl = document.getElementById('loading-detail');
const appEl = document.getElementById('app');
const inputEl = document.getElementById('query-input');
const searchBtn = document.getElementById('search-btn');
const resultsEl = document.getElementById('results');
const dictSizeEl = document.getElementById('dict-size');
const rulesCountEl = document.getElementById('rules-count');
const initTimeEl = document.getElementById('init-time');

// State
let initTime = 0;

/**
 * Initialize the WASM module and spellchecker
 */
async function initialize() {
    const startTime = performance.now();

    try {
        // Step 1: Load WASM module
        loadingDetailEl.textContent = 'Loading WASM module...';
        await init();

        // Step 2: Initialize spellchecker (builds index from embedded data)
        loadingDetailEl.textContent = 'Building search index...';
        await initSpellchecker();

        initTime = performance.now() - startTime;
        console.log(`Spellchecker initialized in ${initTime.toFixed(0)}ms`);

        // Update stats display
        const stats = get_stats();
        dictSizeEl.textContent = stats.dictionary_size.toLocaleString();
        rulesCountEl.textContent = stats.rules_count;
        initTimeEl.textContent = `${initTime.toFixed(0)}ms`;

        // Show app, hide loading
        loadingEl.classList.add('hidden');
        appEl.classList.remove('hidden');
        inputEl.focus();

    } catch (error) {
        console.error('Failed to initialize:', error);
        loadingEl.innerHTML = `
            <p class="error">Failed to load spellchecker</p>
            <p class="error-detail">${error.message || error}</p>
        `;
    }
}

/**
 * Perform a spelling search
 */
function performSearch() {
    const word = inputEl.value.trim().toLowerCase();
    if (!word) {
        resultsEl.innerHTML = '<p class="hint">Enter a word to search</p>';
        return;
    }

    const startTime = performance.now();

    try {
        const result = query(word);
        const queryTime = performance.now() - startTime;

        displayResults(result, queryTime);

    } catch (error) {
        console.error('Query failed:', error);
        resultsEl.innerHTML = `<p class="error">Error: ${error.message || error}</p>`;
    }
}

/**
 * Display search results
 */
function displayResults(result, queryTime) {
    let html = `
        <div class="result-header">
            <div class="result-title">
                <span class="original">"${escapeHtml(result.original)}"</span>
                <span class="arrow">→</span>
                <span class="normalized">"${escapeHtml(result.normalized)}"</span>
            </div>
            <span class="timing">${queryTime.toFixed(1)}ms${result.from_cache ? ' (cached)' : ''}</span>
        </div>
    `;

    if (result.warning) {
        html += `<p class="warning">${escapeHtml(result.warning)}</p>`;
    }

    if (result.matches.length === 0) {
        html += '<p class="no-matches">No matches found within distance 2</p>';
    } else {
        html += '<ol class="match-list">';
        for (const match of result.matches) {
            const distanceClass = `distance-${Math.min(match.distance, 2)}`;
            const exactNote = match.distance === 0
                ? ' <span class="exact-match">(exact phonetic match)</span>'
                : '';
            html += `
                <li class="${distanceClass}">
                    <span class="match-word">${escapeHtml(match.word)}</span>
                    <span class="match-distance">distance: ${match.distance}</span>
                    ${exactNote}
                </li>
            `;
        }
        html += '</ol>';
    }

    resultsEl.innerHTML = html;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Event listeners
searchBtn.addEventListener('click', performSearch);

inputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// Live search on input (with debounce)
let searchTimeout = null;
inputEl.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        if (inputEl.value.trim().length >= 2) {
            performSearch();
        }
    }, 150);
});

// Example buttons
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        inputEl.value = btn.dataset.query;
        performSearch();
    });
});

// Initialize on load
initialize();
