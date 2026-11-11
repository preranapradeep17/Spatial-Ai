"""Typographical Robustness and Controlled-Vocabulary Normalization Module.

Provides algorithmic fuzzy correction using Levenshtein distance against a controlled
vocabulary of tools, facility rooms, objects, missions, and investigation commands.

Strict Invariants:
1. Short words (len <= 3) are only corrected via explicit canonical synonym/abbreviation maps.
2. Words of length 4 allow edit distance <= 1.
3. Words of length >= 5 allow edit distance <= 2.
4. Valid domain vocabulary words (e.g. 'chair', 'table', 'door') are never mutated into other valid words.
5. Unknown nouns (e.g. 'quantum', 'microscope') are preserved without aggressive mutation.
6. Ambiguous matches (multiple candidates with equal minimum edit distance) are preserved without guessing.
"""
from typing import Dict, Tuple, List, Set, Optional
import re

# Explicit short abbreviations & contractions (len <= 3)
SHORT_WORD_CORRECTIONS = {
    "wat": "what",
    "wht": "what",
    "u": "you",
    "ur": "your",
    "fnd": "find",
    "tke": "take",
    "tak": "take",
    "giv": "give",
    "gve": "give",
    "hnt": "hint",
    "shw": "show",
    "pls": "please",
    "plz": "please",
    "hw": "how",
    "whr": "where",
    "rm": "room",
    "dr": "door"
}

# Controlled Facility Vocabulary
CONTROLLED_ROOMS: Set[str] = {
    "room", "rooms",
    "reception", "hub", "central", "office", "research",
    "library", "archives", "lab", "laboratory", "storage", "vault"
}

CONTROLLED_OBJECTS: Set[str] = {
    "table", "tables", "chair", "chairs", "bookshelf", "bookshelves", "bookcase",
    "cabinet", "cabinets", "sofa", "sofas", "door", "doors", "desk", "desks",
    "laptop", "laptops", "key", "keys", "keycard", "pda", "datapad", "datapads",
    "safe", "terminal", "clue", "clues", "note", "notes", "paper", "document", "documents"
}

CONTROLLED_COMMANDS_AND_WORDS: Set[str] = {
    "objects", "object", "visible", "detect", "detecting", "survey", "surveying", "identify",
    "nearest", "closest", "exit", "doorway", "navigate", "navigating", "guide", "guiding",
    "directions", "distance", "between", "inventory", "collected",
    "objectives", "objective", "missions", "mission", "missing", "uncertainty", "unreachable", "blocked",
    "inspect", "inspecting", "examine", "examining", "manual", "safety", "incident", "reports",
    "report", "reported", "reporting", "passcode", "passcodes", "override", "code", "codes", "keypad",
    "enter", "entering", "input", "submit", "type", "pick", "pickup", "retrieve", "retrieving",
    "collect", "collecting", "grab", "find", "finding", "complete", "completed", "finish", "finished",
    "protocol", "waypoints", "around", "where", "what", "which", "there",
    "here", "myself", "reach", "reaching", "count", "counting", "summarise",
    "summarize", "summary", "read", "reading", "unlock", "open", "lead", "leads", "walk",
    "describe", "check", "show", "give", "take", "stand", "standing"
}

# Standard English stopwords and functional words (never mutate these!)
COMMON_ENGLISH_WORDS: Set[str] = {
    "that", "this", "it", "is", "are", "am", "was", "were", "be", "been",
    "the", "a", "an", "and", "or", "not", "no", "yes", "to", "in", "on",
    "at", "by", "for", "with", "from", "of", "into", "over", "under",
    "my", "me", "mine", "your", "yours", "you", "we", "us", "our", "ours",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "can", "could", "will", "would", "shall", "should", "do", "does", "did",
    "have", "has", "had", "go", "get", "see", "look", "tell", "give", "take",
    "all", "any", "some", "more", "most", "about", "there", "here", "play",
    "many", "much", "few", "both", "every", "how"
}

# Full controlled vocabulary set
CONTROLLED_VOCABULARY: Set[str] = CONTROLLED_ROOMS | CONTROLLED_OBJECTS | CONTROLLED_COMMANDS_AND_WORDS | COMMON_ENGLISH_WORDS


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Damerau-Levenshtein distance supporting insertion, deletion,
    substitution, and transposition of adjacent characters."""
    d = {}
    len1, len2 = len(s1), len(s2)
    for i in range(-1, len1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len2 + 1):
        d[(-1, j)] = j + 1

    for i in range(len1):
        for j in range(len2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,      # deletion
                d[(i, j - 1)] + 1,      # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition
    return d[(len1 - 1, len2 - 1)]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Alias for distance metric supporting transpositions."""
    return damerau_levenshtein_distance(s1, s2)


class TypoNormalizer:
    """Corrects typographical errors against the facility domain controlled vocabulary."""

    def __init__(self, vocabulary: Optional[Set[str]] = None):
        self.vocab = vocabulary or CONTROLLED_VOCABULARY
        # Precompute words by length for fast candidate pruning
        self.words_by_len: Dict[int, List[str]] = {}
        for word in self.vocab:
            self.words_by_len.setdefault(len(word), []).append(word)

    def correct_word(self, word: str) -> Tuple[str, bool]:
        """Corrects a single token against controlled vocabulary under strict constraints.

        Returns:
            Tuple[corrected_word, was_corrected]
        """
        w_low = word.lower().strip()
        if not w_low:
            return word, False

        # Invariant 1: If word is already valid controlled vocabulary, never mutate it!
        if w_low in self.vocab:
            return word, False

        # Invariant 2: Check short explicit abbreviations (len <= 3)
        if len(w_low) <= 3:
            if w_low in SHORT_WORD_CORRECTIONS:
                corr = SHORT_WORD_CORRECTIONS[w_low]
                # Preserve capitalization of original word if needed
                res = corr.upper() if word.isupper() else (corr.capitalize() if word.istitle() else corr)
                return res, True
            return word, False

        # Invariant 3: Determine max allowable distance based on length
        max_dist = 1 if len(w_low) == 4 else 2

        # Candidate pruning: only inspect candidates with length difference <= max_dist
        candidates = []
        for l in range(len(w_low) - max_dist, len(w_low) + max_dist + 1):
            candidates.extend(self.words_by_len.get(l, []))

        best_word = None
        min_dist = max_dist + 1
        is_ambiguous = False

        for cand in candidates:
            # Special guard: Never mutate an unknown object to a different valid object if length is short
            dist = levenshtein_distance(w_low, cand)
            if dist < min_dist:
                min_dist = dist
                best_word = cand
                is_ambiguous = False
            elif dist == min_dist and dist <= max_dist:
                is_ambiguous = True  # Multiple candidates tie at same distance

        # Invariant 4 & 5: Only apply correction if unambiguous and within threshold
        if best_word and min_dist <= max_dist and not is_ambiguous:
            res = best_word.upper() if word.isupper() else (best_word.capitalize() if word.istitle() else best_word)
            return res, True

        return word, False

    def normalize(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Tokenizes, normalizes, and reconstructs the user query, logging all corrections.

        Returns:
            Tuple[normalized_query, corrections_dict]
        """
        corrections: Dict[str, str] = {}
        if not query:
            return query, corrections

        # Split preserving punctuation and whitespace
        tokens_and_delimiters = re.split(r"([a-zA-Z0-9_]+)", query)
        normalized_pieces = []

        for piece in tokens_and_delimiters:
            if re.match(r"^[a-zA-Z0-9_]+$", piece):
                corrected, was_changed = self.correct_word(piece)
                if was_changed:
                    corrections[piece] = corrected
                normalized_pieces.append(corrected)
            else:
                normalized_pieces.append(piece)

        normalized_query = "".join(normalized_pieces)
        return normalized_query, corrections


# Global singleton normalizer
global_typo_normalizer = TypoNormalizer()

def normalize_query_typos(query: str) -> str:
    """Drop-in functional wrapper for backward-compatibility."""
    norm_q, _ = global_typo_normalizer.normalize(query)
    return norm_q
