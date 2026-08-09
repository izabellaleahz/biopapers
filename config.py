"""biopapers MVP v0 — configuration.

Lab subfields mapped onto bioRxiv subject categories. The bioRxiv `details`
API returns a `category` per preprint; we bucket the feed on these.
Edit freely as the lab's interests shift.
"""

# Each topic: a URL slug -> display label + the bioRxiv categories it includes.
# bioRxiv categories are matched case-insensitively.
# `color` is a pastel accent per topic — used for pills + card accents in the UI.
TOPICS = {
    "cancer": {
        "label": "Cancer Biology",
        "categories": ["cancer biology"],
        "color": "#e88aa6",  # rose
    },
    "immunology": {
        "label": "Immunology / T cell",
        "categories": ["immunology"],
        "color": "#a98fd6",  # lavender
    },
    "microbiology": {
        # Virology lives under Microbiology on bioRxiv.
        "label": "Micro & Viral",
        "categories": ["microbiology"],
        "color": "#5cc0a0",  # mint
    },
    "molecular": {
        "label": "Molecular Biology",
        "categories": ["molecular biology", "biochemistry"],
        "color": "#eea777",  # peach
    },
    "bioml": {
        # Gilbert-lab-flavored computational bucket.
        "label": "Bio ML",
        "categories": ["bioinformatics", "systems biology"],
        "color": "#6bb6e6",  # sky
    },
}

# Flat set of every category we care about (union of all topics), lowercased.
INTERESTING_CATEGORIES = {
    cat.lower()
    for topic in TOPICS.values()
    for cat in topic["categories"]
}


def topic_for_category(category: str) -> str | None:
    """Return the topic slug a bioRxiv category belongs to, or None."""
    if not category:
        return None
    cat = category.lower().strip()
    for slug, topic in TOPICS.items():
        if cat in (c.lower() for c in topic["categories"]):
            return slug
    return None
