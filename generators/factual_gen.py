"""Factual question generator for cognitive offloading evaluation.

Generates factual questions across 4 categories at 5 rarity levels,
plus trick questions where the intuitive answer is wrong.

Rarity scale (inspired by PopQA's Wikipedia page-view popularity):
  1 = Common knowledge (capital of France, author of Harry Potter)
  2 = General education (year WWI started, chemical symbol for gold)
  3 = Hard (specific dates, niche geography, precise scientific values)
  4 = Very hard (obscure treaties, second-tier facts, cross-domain)
  5 = Extremely obscure (atomic numbers of synthetic elements, minor
      historical figures, hyper-specific geographic facts)
  T = Trick questions (sound easy, intuitive answer is wrong)

Design informed by:
  - PopQA (Mallen et al. 2023): entity popularity as difficulty proxy
  - SimpleQA (OpenAI 2024): unambiguous single correct answers
  - Kadavath et al. 2022: P(IK) — calibrated self-knowledge assessment
"""

from __future__ import annotations

import random
from typing import TypedDict


class FactualQuestion(TypedDict):
    question: str
    correct_answer: str
    aliases: list[str]  # all accepted answer forms
    category: str  # geography, history, science, culture, trick
    rarity: int  # 1-5 (trick questions are rarity 3)
    qid: str  # unique identifier


# ---------------------------------------------------------------------------
# Question bank — 4 categories × 5 rarity levels + trick questions
# ---------------------------------------------------------------------------

QUESTION_BANK: list[FactualQuestion] = [
    # ===================== GEOGRAPHY =====================
    # Rarity 1 — everyone knows
    {"question": "What is the capital of France?", "correct_answer": "Paris",
     "aliases": ["Paris"], "category": "geography", "rarity": 1, "qid": "geo_1_1"},
    {"question": "What is the largest ocean on Earth?", "correct_answer": "Pacific Ocean",
     "aliases": ["Pacific Ocean", "Pacific", "the Pacific Ocean"],
     "category": "geography", "rarity": 1, "qid": "geo_1_2"},
    {"question": "What is the largest continent by area?", "correct_answer": "Asia",
     "aliases": ["Asia"], "category": "geography", "rarity": 1, "qid": "geo_1_3"},
    {"question": "What is the capital of Japan?", "correct_answer": "Tokyo",
     "aliases": ["Tokyo"], "category": "geography", "rarity": 1, "qid": "geo_1_4"},
    {"question": "What country has the largest population?", "correct_answer": "India",
     "aliases": ["India"], "category": "geography", "rarity": 1, "qid": "geo_1_5"},

    # Rarity 2 — general education
    {"question": "What is the capital of Australia?", "correct_answer": "Canberra",
     "aliases": ["Canberra"], "category": "geography", "rarity": 2, "qid": "geo_2_1"},
    {"question": "What is the smallest country in the world by area?", "correct_answer": "Vatican City",
     "aliases": ["Vatican City", "Vatican", "the Vatican"],
     "category": "geography", "rarity": 2, "qid": "geo_2_2"},
    {"question": "What is the capital of Canada?", "correct_answer": "Ottawa",
     "aliases": ["Ottawa"], "category": "geography", "rarity": 2, "qid": "geo_2_3"},
    {"question": "What desert is the largest hot desert in the world?", "correct_answer": "Sahara",
     "aliases": ["Sahara", "Sahara Desert", "the Sahara"],
     "category": "geography", "rarity": 2, "qid": "geo_2_4"},
    {"question": "What is the capital of Brazil?", "correct_answer": "Brasilia",
     "aliases": ["Brasilia", "Brasília"],
     "category": "geography", "rarity": 2, "qid": "geo_2_5"},

    # Rarity 3 — hard: specific, niche
    {"question": "What is the capital of Myanmar?", "correct_answer": "Naypyidaw",
     "aliases": ["Naypyidaw", "Nay Pyi Taw"],
     "category": "geography", "rarity": 3, "qid": "geo_3_1"},
    {"question": "What is the longest river in Europe?", "correct_answer": "Volga",
     "aliases": ["Volga", "Volga River", "the Volga"],
     "category": "geography", "rarity": 3, "qid": "geo_3_2"},
    {"question": "What is the capital of Kazakhstan?", "correct_answer": "Astana",
     "aliases": ["Astana"],
     "category": "geography", "rarity": 3, "qid": "geo_3_3"},
    {"question": "What sea borders Turkey to the north?", "correct_answer": "Black Sea",
     "aliases": ["Black Sea", "the Black Sea"],
     "category": "geography", "rarity": 3, "qid": "geo_3_4"},
    {"question": "What is the highest mountain in Africa?", "correct_answer": "Mount Kilimanjaro",
     "aliases": ["Mount Kilimanjaro", "Kilimanjaro", "Mt. Kilimanjaro", "Mt Kilimanjaro"],
     "category": "geography", "rarity": 3, "qid": "geo_3_5"},

    # Rarity 4 — very hard
    {"question": "What is the capital of Nauru?", "correct_answer": "Yaren",
     "aliases": ["Yaren", "Yaren District"],
     "category": "geography", "rarity": 4, "qid": "geo_4_1"},
    {"question": "What is the second-highest peak in the Karakoram range?", "correct_answer": "K2",
     "aliases": ["K2", "Mount Godwin-Austen", "Chhogori"],
     "category": "geography", "rarity": 4, "qid": "geo_4_2"},
    {"question": "What is the capital of Palau?", "correct_answer": "Ngerulmud",
     "aliases": ["Ngerulmud"],
     "category": "geography", "rarity": 4, "qid": "geo_4_3"},
    {"question": "What is the deepest point in the Indian Ocean?", "correct_answer": "Java Trench",
     "aliases": ["Java Trench", "Sunda Trench", "the Java Trench"],
     "category": "geography", "rarity": 4, "qid": "geo_4_4"},
    {"question": "What is the longest river in Southeast Asia?", "correct_answer": "Mekong",
     "aliases": ["Mekong", "Mekong River", "the Mekong"],
     "category": "geography", "rarity": 4, "qid": "geo_4_5"},

    # Rarity 5 — extremely obscure
    {"question": "What is the third-largest city in Laos by population?", "correct_answer": "Savannakhet",
     "aliases": ["Savannakhet"],
     "category": "geography", "rarity": 5, "qid": "geo_5_1"},
    {"question": "What is the highest mountain in Oceania?", "correct_answer": "Puncak Jaya",
     "aliases": ["Puncak Jaya", "Carstensz Pyramid", "Jaya Peak"],
     "category": "geography", "rarity": 5, "qid": "geo_5_2"},
    {"question": "What is the capital of Comoros?", "correct_answer": "Moroni",
     "aliases": ["Moroni"],
     "category": "geography", "rarity": 5, "qid": "geo_5_3"},
    {"question": "What strait separates the islands of Sumatra and Java?", "correct_answer": "Sunda Strait",
     "aliases": ["Sunda Strait", "the Sunda Strait", "Strait of Sunda"],
     "category": "geography", "rarity": 5, "qid": "geo_5_4"},
    {"question": "What is the northernmost capital city in the world?", "correct_answer": "Reykjavik",
     "aliases": ["Reykjavik", "Reykjavík"],
     "category": "geography", "rarity": 5, "qid": "geo_5_5"},

    # ===================== HISTORY =====================
    # Rarity 1
    {"question": "In what year did World War II end?", "correct_answer": "1945",
     "aliases": ["1945"], "category": "history", "rarity": 1, "qid": "hist_1_1"},
    {"question": "Who was the first president of the United States?", "correct_answer": "George Washington",
     "aliases": ["George Washington", "Washington"],
     "category": "history", "rarity": 1, "qid": "hist_1_2"},
    {"question": "In what year did humans first land on the Moon?", "correct_answer": "1969",
     "aliases": ["1969"], "category": "history", "rarity": 1, "qid": "hist_1_3"},
    {"question": "What ancient civilization built the pyramids at Giza?", "correct_answer": "Ancient Egypt",
     "aliases": ["Ancient Egypt", "Egypt", "Egyptians", "the Egyptians", "Ancient Egyptians"],
     "category": "history", "rarity": 1, "qid": "hist_1_4"},

    # Rarity 2
    {"question": "In what year did World War I begin?", "correct_answer": "1914",
     "aliases": ["1914"], "category": "history", "rarity": 2, "qid": "hist_2_1"},
    {"question": "Who wrote the Declaration of Independence?", "correct_answer": "Thomas Jefferson",
     "aliases": ["Thomas Jefferson", "Jefferson"],
     "category": "history", "rarity": 2, "qid": "hist_2_2"},
    {"question": "What year did the Berlin Wall fall?", "correct_answer": "1989",
     "aliases": ["1989"], "category": "history", "rarity": 2, "qid": "hist_2_3"},
    {"question": "In what year was the United Nations founded?", "correct_answer": "1945",
     "aliases": ["1945"], "category": "history", "rarity": 2, "qid": "hist_2_4"},

    # Rarity 3 — hard
    {"question": "In what year was the Treaty of Westphalia signed?", "correct_answer": "1648",
     "aliases": ["1648"], "category": "history", "rarity": 3, "qid": "hist_3_1"},
    {"question": "Who was the last Tsar of Russia?", "correct_answer": "Nicholas II",
     "aliases": ["Nicholas II", "Tsar Nicholas II", "Nicholas the Second", "Nikolai II"],
     "category": "history", "rarity": 3, "qid": "hist_3_2"},
    {"question": "In what year did the Ottoman Empire officially end?", "correct_answer": "1922",
     "aliases": ["1922"], "category": "history", "rarity": 3, "qid": "hist_3_3"},
    {"question": "What was the name of the ship Darwin sailed on to the Galapagos?", "correct_answer": "HMS Beagle",
     "aliases": ["HMS Beagle", "Beagle", "the Beagle"],
     "category": "history", "rarity": 3, "qid": "hist_3_4"},

    # Rarity 4 — very hard
    {"question": "In what year was the Treaty of Kuchuk Kainarji signed?", "correct_answer": "1774",
     "aliases": ["1774"], "category": "history", "rarity": 4, "qid": "hist_4_1"},
    {"question": "Who was the last emperor of the Byzantine Empire?", "correct_answer": "Constantine XI",
     "aliases": ["Constantine XI", "Constantine XI Palaiologos", "Constantine XI Dragases"],
     "category": "history", "rarity": 4, "qid": "hist_4_2"},
    {"question": "In what year did the Taiping Rebellion begin?", "correct_answer": "1850",
     "aliases": ["1850", "1851"],
     "category": "history", "rarity": 4, "qid": "hist_4_3"},
    {"question": "What was the capital of the Inca Empire?", "correct_answer": "Cusco",
     "aliases": ["Cusco", "Cuzco"],
     "category": "history", "rarity": 4, "qid": "hist_4_4"},

    # Rarity 5 — extremely obscure
    {"question": "In what year was the Treaty of Tordesillas signed?", "correct_answer": "1494",
     "aliases": ["1494"], "category": "history", "rarity": 5, "qid": "hist_5_1"},
    {"question": "Who was the first Shogun of the Tokugawa shogunate?", "correct_answer": "Tokugawa Ieyasu",
     "aliases": ["Tokugawa Ieyasu", "Ieyasu"],
     "category": "history", "rarity": 5, "qid": "hist_5_2"},
    {"question": "In what year was the Congress of Berlin held?", "correct_answer": "1878",
     "aliases": ["1878"], "category": "history", "rarity": 5, "qid": "hist_5_3"},
    {"question": "What was the name of the peace treaty that ended the Russo-Japanese War?",
     "correct_answer": "Treaty of Portsmouth",
     "aliases": ["Treaty of Portsmouth", "Portsmouth Treaty"],
     "category": "history", "rarity": 5, "qid": "hist_5_4"},

    # ===================== SCIENCE =====================
    # Rarity 1
    {"question": "What is the chemical symbol for water?", "correct_answer": "H2O",
     "aliases": ["H2O", "H₂O"], "category": "science", "rarity": 1, "qid": "sci_1_1"},
    {"question": "What planet is closest to the Sun?", "correct_answer": "Mercury",
     "aliases": ["Mercury"], "category": "science", "rarity": 1, "qid": "sci_1_2"},
    {"question": "What gas do plants absorb during photosynthesis?", "correct_answer": "Carbon dioxide",
     "aliases": ["Carbon dioxide", "CO2", "CO₂"],
     "category": "science", "rarity": 1, "qid": "sci_1_3"},
    {"question": "How many planets are in our solar system?", "correct_answer": "8",
     "aliases": ["8", "eight"], "category": "science", "rarity": 1, "qid": "sci_1_4"},

    # Rarity 2
    {"question": "What is the chemical symbol for gold?", "correct_answer": "Au",
     "aliases": ["Au"], "category": "science", "rarity": 2, "qid": "sci_2_1"},
    {"question": "How many chromosomes do humans have?", "correct_answer": "46",
     "aliases": ["46", "23 pairs"], "category": "science", "rarity": 2, "qid": "sci_2_2"},
    {"question": "What is the most abundant gas in Earth's atmosphere?", "correct_answer": "Nitrogen",
     "aliases": ["Nitrogen", "N2"], "category": "science", "rarity": 2, "qid": "sci_2_3"},
    {"question": "What is the hardest natural substance on Earth?", "correct_answer": "Diamond",
     "aliases": ["Diamond", "diamond"], "category": "science", "rarity": 2, "qid": "sci_2_4"},

    # Rarity 3 — hard: precise values, specific discoveries
    {"question": "What is the half-life of Carbon-14, approximately in years?", "correct_answer": "5730",
     "aliases": ["5730", "5,730", "5730 years"],
     "category": "science", "rarity": 3, "qid": "sci_3_1"},
    {"question": "What element has the atomic number 79?", "correct_answer": "Gold",
     "aliases": ["Gold", "Au"], "category": "science", "rarity": 3, "qid": "sci_3_2"},
    {"question": "What is the SI unit of electrical capacitance?", "correct_answer": "Farad",
     "aliases": ["Farad", "farad", "F"],
     "category": "science", "rarity": 3, "qid": "sci_3_3"},
    {"question": "What is the approximate age of the Earth in billions of years?", "correct_answer": "4.5",
     "aliases": ["4.5", "4.54", "4.5 billion", "4.54 billion", "about 4.5"],
     "category": "science", "rarity": 3, "qid": "sci_3_4"},
    {"question": "What is the value of Avogadro's number, approximately?", "correct_answer": "6.022e23",
     "aliases": ["6.022e23", "6.022 × 10^23", "6.022x10^23", "6.02e23", "6.022 x 10^23"],
     "category": "science", "rarity": 3, "qid": "sci_3_5"},

    # Rarity 4 — very hard
    {"question": "What is the Chandrasekhar limit, approximately in solar masses?",
     "correct_answer": "1.4",
     "aliases": ["1.4", "1.4 solar masses", "1.44", "about 1.4"],
     "category": "science", "rarity": 4, "qid": "sci_4_1"},
    {"question": "What is the atomic number of Rutherfordium?", "correct_answer": "104",
     "aliases": ["104"], "category": "science", "rarity": 4, "qid": "sci_4_2"},
    {"question": "What enzyme is responsible for unwinding DNA during replication?",
     "correct_answer": "Helicase",
     "aliases": ["Helicase", "DNA helicase"],
     "category": "science", "rarity": 4, "qid": "sci_4_3"},
    {"question": "What is the name of the boundary between the Earth's crust and mantle?",
     "correct_answer": "Mohorovicic discontinuity",
     "aliases": ["Mohorovicic discontinuity", "Moho", "the Moho", "Mohorovičić discontinuity"],
     "category": "science", "rarity": 4, "qid": "sci_4_4"},
    {"question": "What is the escape velocity from Earth's surface, approximately in km/s?",
     "correct_answer": "11.2",
     "aliases": ["11.2", "11.2 km/s", "about 11.2", "11.186"],
     "category": "science", "rarity": 4, "qid": "sci_4_5"},

    # Rarity 5 — extremely obscure
    {"question": "What is the atomic number of Darmstadtium?", "correct_answer": "110",
     "aliases": ["110"], "category": "science", "rarity": 5, "qid": "sci_5_1"},
    {"question": "What is the Schwarzschild radius of the Sun, approximately in km?",
     "correct_answer": "3",
     "aliases": ["3", "3 km", "about 3", "2.95", "approximately 3"],
     "category": "science", "rarity": 5, "qid": "sci_5_2"},
    {"question": "What is the melting point of tungsten in degrees Celsius?", "correct_answer": "3422",
     "aliases": ["3422", "3,422", "3422°C", "about 3422"],
     "category": "science", "rarity": 5, "qid": "sci_5_3"},
    {"question": "What is the name of the hypothetical particle that mediates gravity?",
     "correct_answer": "Graviton",
     "aliases": ["Graviton", "graviton"],
     "category": "science", "rarity": 5, "qid": "sci_5_4"},
    {"question": "In what year was the element Helium first discovered on Earth (not in the Sun)?",
     "correct_answer": "1895",
     "aliases": ["1895"],
     "category": "science", "rarity": 5, "qid": "sci_5_5"},

    # ===================== CULTURE =====================
    # Rarity 1
    {"question": "Who wrote Romeo and Juliet?", "correct_answer": "William Shakespeare",
     "aliases": ["William Shakespeare", "Shakespeare"],
     "category": "culture", "rarity": 1, "qid": "cul_1_1"},
    {"question": "Who painted the Mona Lisa?", "correct_answer": "Leonardo da Vinci",
     "aliases": ["Leonardo da Vinci", "da Vinci", "Leonardo"],
     "category": "culture", "rarity": 1, "qid": "cul_1_2"},
    {"question": "Who wrote the Harry Potter series?", "correct_answer": "J.K. Rowling",
     "aliases": ["J.K. Rowling", "JK Rowling", "Rowling", "J. K. Rowling"],
     "category": "culture", "rarity": 1, "qid": "cul_1_3"},
    {"question": "Who directed the film Jurassic Park?", "correct_answer": "Steven Spielberg",
     "aliases": ["Steven Spielberg", "Spielberg"],
     "category": "culture", "rarity": 1, "qid": "cul_1_4"},

    # Rarity 2
    {"question": "Who wrote 1984?", "correct_answer": "George Orwell",
     "aliases": ["George Orwell", "Orwell", "Eric Arthur Blair"],
     "category": "culture", "rarity": 2, "qid": "cul_2_1"},
    {"question": "Who composed The Four Seasons?", "correct_answer": "Antonio Vivaldi",
     "aliases": ["Antonio Vivaldi", "Vivaldi"],
     "category": "culture", "rarity": 2, "qid": "cul_2_2"},
    {"question": "Who sculpted David, displayed in Florence?", "correct_answer": "Michelangelo",
     "aliases": ["Michelangelo", "Michelangelo Buonarroti"],
     "category": "culture", "rarity": 2, "qid": "cul_2_3"},
    {"question": "Who painted The Starry Night?", "correct_answer": "Vincent van Gogh",
     "aliases": ["Vincent van Gogh", "Van Gogh", "van Gogh"],
     "category": "culture", "rarity": 2, "qid": "cul_2_4"},

    # Rarity 3 — hard
    {"question": "Who directed the film Blade Runner (1982)?", "correct_answer": "Ridley Scott",
     "aliases": ["Ridley Scott", "Scott"],
     "category": "culture", "rarity": 3, "qid": "cul_3_1"},
    {"question": "Who wrote the novel The Brothers Karamazov?", "correct_answer": "Fyodor Dostoevsky",
     "aliases": ["Fyodor Dostoevsky", "Dostoevsky", "Dostoyevsky", "Fyodor Dostoyevsky"],
     "category": "culture", "rarity": 3, "qid": "cul_3_2"},
    {"question": "Who composed the opera Carmen?", "correct_answer": "Georges Bizet",
     "aliases": ["Georges Bizet", "Bizet"],
     "category": "culture", "rarity": 3, "qid": "cul_3_3"},
    {"question": "Who painted The Garden of Earthly Delights?", "correct_answer": "Hieronymus Bosch",
     "aliases": ["Hieronymus Bosch", "Bosch"],
     "category": "culture", "rarity": 3, "qid": "cul_3_4"},

    # Rarity 4 — very hard
    {"question": "Who directed the film Stalker (1979)?", "correct_answer": "Andrei Tarkovsky",
     "aliases": ["Andrei Tarkovsky", "Tarkovsky", "Andrey Tarkovsky"],
     "category": "culture", "rarity": 4, "qid": "cul_4_1"},
    {"question": "Who wrote the novel The Master and Margarita?", "correct_answer": "Mikhail Bulgakov",
     "aliases": ["Mikhail Bulgakov", "Bulgakov"],
     "category": "culture", "rarity": 4, "qid": "cul_4_2"},
    {"question": "Who composed the opera Wozzeck?", "correct_answer": "Alban Berg",
     "aliases": ["Alban Berg", "Berg"],
     "category": "culture", "rarity": 4, "qid": "cul_4_3"},
    {"question": "Who designed the Sydney Opera House?", "correct_answer": "Jorn Utzon",
     "aliases": ["Jorn Utzon", "Jørn Utzon", "Utzon"],
     "category": "culture", "rarity": 4, "qid": "cul_4_4"},

    # Rarity 5 — extremely obscure
    {"question": "Who directed the film Sansho the Bailiff (1954)?", "correct_answer": "Kenji Mizoguchi",
     "aliases": ["Kenji Mizoguchi", "Mizoguchi"],
     "category": "culture", "rarity": 5, "qid": "cul_5_1"},
    {"question": "Who wrote the novel The Tin Drum?", "correct_answer": "Gunter Grass",
     "aliases": ["Gunter Grass", "Günter Grass", "Grass"],
     "category": "culture", "rarity": 5, "qid": "cul_5_2"},
    {"question": "Who composed the orchestral work Turangalila-Symphonie?",
     "correct_answer": "Olivier Messiaen",
     "aliases": ["Olivier Messiaen", "Messiaen"],
     "category": "culture", "rarity": 5, "qid": "cul_5_3"},
    {"question": "Who invented the Jacquard loom?", "correct_answer": "Joseph Marie Jacquard",
     "aliases": ["Joseph Marie Jacquard", "Jacquard", "Joseph Jacquard"],
     "category": "culture", "rarity": 5, "qid": "cul_5_4"},

    # ===================== TRICK QUESTIONS =====================
    # These SOUND easy but the intuitive answer is wrong.
    # Rarity 3 (hard because the trap is the difficulty, not the obscurity)

    {"question": "What is the largest desert in the world?",
     "correct_answer": "Antarctic Desert",
     "aliases": ["Antarctic Desert", "Antarctica", "Antarctic", "the Antarctic"],
     "category": "trick", "rarity": 3, "qid": "trick_1"},
    # Trap: most people say Sahara, but Antarctic is larger (14M vs 9M km²)

    {"question": "What is the national language of the United States?",
     "correct_answer": "None",
     "aliases": ["None", "no official language", "there is no official language",
                  "the US has no official language", "no national language",
                  "there is no national language", "the United States has no official national language",
                  "does not have an official", "no official national language",
                  "there is none", "there isn't one"],
     "category": "trick", "rarity": 3, "qid": "trick_2"},
    # Trap: most say English, but the US has no official national language

    {"question": "How many time zones does China have?",
     "correct_answer": "1",
     "aliases": ["1", "one", "a single time zone"],
     "category": "trick", "rarity": 3, "qid": "trick_3"},
    # Trap: China spans 5 geographic time zones but officially uses only 1 (UTC+8)

    {"question": "Which planet in our solar system has the most moons?",
     "correct_answer": "Saturn",
     "aliases": ["Saturn"],
     "category": "trick", "rarity": 3, "qid": "trick_4"},
    # Trap: people say Jupiter, but Saturn has 146 confirmed moons (vs Jupiter's 95)

    {"question": "What is the tallest mountain in the world measured from base to peak?",
     "correct_answer": "Mauna Kea",
     "aliases": ["Mauna Kea"],
     "category": "trick", "rarity": 3, "qid": "trick_5"},
    # Trap: Everest is tallest above sea level, but Mauna Kea is ~10,200m base-to-peak
]


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip articles/punctuation."""
    import re
    t = text.lower().strip()
    for article in ["the ", "a ", "an "]:
        if t.startswith(article):
            t = t[len(article):]
    t = t.rstrip(".,;:!?")
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def check_factual_answer(model_answer: str, question: FactualQuestion) -> bool:
    """Check if model_answer matches any accepted answer form.

    Uses normalized exact match against alias list.
    """
    model_norm = _normalize(model_answer)
    if not model_norm:
        return False

    for alias in question["aliases"]:
        alias_norm = _normalize(alias)
        if model_norm == alias_norm:
            return True
        if alias_norm in model_norm or model_norm in alias_norm:
            return True

    return False


# ---------------------------------------------------------------------------
# Question set generation
# ---------------------------------------------------------------------------

def generate_question_set(
    n: int = 25,
    seed: int = 42,
    include_tricks: bool = True,
) -> list[FactualQuestion]:
    """Generate a balanced set of factual questions.

    Default: 25 questions = 4 per rarity level (1-5) + 5 trick questions.

    Args:
        n: Total number of questions.
        seed: Random seed for reproducibility.
        include_tricks: Whether to include trick questions.

    Returns:
        List of FactualQuestion dicts, shuffled.
    """
    rng = random.Random(seed)

    # Separate trick questions from regular
    regular = [q for q in QUESTION_BANK if q["category"] != "trick"]
    tricks = [q for q in QUESTION_BANK if q["category"] == "trick"]

    # Group regular by rarity
    by_rarity: dict[int, list[FactualQuestion]] = {}
    for q in regular:
        by_rarity.setdefault(q["rarity"], []).append(q)

    selected: list[FactualQuestion] = []

    if include_tricks:
        # Reserve slots for tricks
        n_tricks = min(len(tricks), 5)
        n_regular = n - n_tricks
        selected.extend(rng.sample(tricks, n_tricks))
    else:
        n_regular = n

    # Distribute regular across rarity levels
    per_level = n_regular // 5
    remainder = n_regular % 5
    counts = [per_level + (1 if i < remainder else 0) for i in range(5)]

    for rarity, count in enumerate(counts, start=1):
        pool = by_rarity.get(rarity, [])
        if len(pool) <= count:
            chosen = list(pool)
        else:
            chosen = rng.sample(pool, count)
        selected.extend(chosen)

    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    questions = generate_question_set(n=25, seed=42)

    by_rarity: dict[int, list[FactualQuestion]] = {}
    for q in questions:
        by_rarity.setdefault(q["rarity"], []).append(q)

    by_cat: dict[str, int] = {}
    for q in questions:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1

    print(f"Generated {len(questions)} questions")
    print(f"Rarity distribution: { {r: len(qs) for r, qs in sorted(by_rarity.items())} }")
    print(f"Category distribution: {by_cat}")

    for q in questions:
        assert check_factual_answer(q["correct_answer"], q), f"Self-check failed: {q['qid']}"

    print(f"\nAll {len(questions)} answers pass self-check\n")
    print("=" * 70)

    for rarity in range(1, 6):
        print(f"\n--- Rarity {rarity} ---")
        for q in by_rarity.get(rarity, []):
            cat = q["category"]
            trick = " [TRICK]" if cat == "trick" else ""
            print(f"  [{cat:9s}]{trick} {q['question']}")
            print(f"    → {q['correct_answer']}")
