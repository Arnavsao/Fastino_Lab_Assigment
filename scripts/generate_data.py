"""
Synthetic Data Generation Script (Bonus).

Generates diverse zero-shot classification training data using the OpenAI API.
Falls back to a template-based method if no API key is available.

Usage:
    # With OpenAI API:
    export OPENAI_API_KEY=your-key
    python scripts/generate_data.py --num_samples 500 --output data/synthetic_data.json

    # Without OpenAI (template-based fallback):
    python scripts/generate_data.py --no_api --num_samples 500 --output data/synthetic_data.json
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Dict

# ---------------------------------------------------------------------------
# Template-based generation (no API required)
# ---------------------------------------------------------------------------

TEMPLATES = {
    "Finance": [
        "The stock market {verb} {percent}% amid {cause}.",
        "{Company} reported {adj} quarterly earnings, beating analyst expectations.",
        "Interest rates {action} as the central bank responds to {condition}.",
        "Investors {react} after {event} shook financial markets globally.",
    ],
    "Technology": [
        "{Company} unveiled a new {product} with groundbreaking {feature}.",
        "Researchers developed an {adj} algorithm that {achievement}.",
        "{Tech} adoption surged {percent}% year-over-year across industries.",
        "The latest {device} sets a record for {metric}, reshaping the market.",
    ],
    "Health": [
        "A new study links {food} consumption to {effect} in adults.",
        "Researchers identified a biomarker for early {disease} detection.",
        "{Treatment} reduces {symptom} in {percent}% of trial participants.",
        "Hospitals adopt {technology} to improve {outcome} for patients.",
    ],
    "Environment": [
        "Climate scientists warn of accelerating {phenomenon} in {region}.",
        "A new {initiative} aims to reduce {pollutant} emissions by {percent}%.",
        "{Species} population {trend} as habitat destruction continues.",
        "Extreme weather events increase {percent}% due to climate change.",
    ],
    "Sports": [
        "{Athlete} breaks the world record in {event} with a time of {time}.",
        "The {team} won the championship after a thrilling {score} victory.",
        "{League} announces expansion to {number} new cities next season.",
        "Sports federation bans {player} for {duration} following doping violation.",
    ],
    "Science": [
        "Scientists discover {phenomenon} that challenges existing {theory}.",
        "A new {material} exhibits {property} at room temperature.",
        "Researchers map the complete {structure} of {organism} for the first time.",
        "The {telescope} captures the most detailed image of {celestial_object} yet.",
    ],
    "Politics": [
        "Parliament passes landmark {bill} with {margin} majority.",
        "Leaders from {number} nations sign {agreement} at the {summit}.",
        "Opposition calls for {action} following {controversy} scandal.",
        "Election results show {party} winning with {percent}% of the popular vote.",
    ],
    "AI": [
        "New language model achieves {score} on {benchmark}, surpassing humans.",
        "Researchers propose {technique} to improve AI {capability}.",
        "{Company} open-sources {model} trained on {data_size} tokens.",
        "AI system {achievement} in {domain}, raising both hopes and concerns.",
    ],
}

FILLERS = {
    "verb": ["surged", "plummeted", "recovered", "stagnated", "rallied"],
    "percent": ["5", "12", "18", "23", "35", "47", "60"],
    "cause": ["inflation fears", "earnings reports", "geopolitical tensions", "policy changes"],
    "Company": ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta"],
    "adj": ["record", "strong", "disappointing", "unexpected", "stellar"],
    "action": ["held steady", "cut", "raised", "paused", "pivoted"],
    "condition": ["rising inflation", "strong employment", "economic slowdown"],
    "react": ["fled", "flooded in", "remained cautious"],
    "event": ["a banking crisis", "a trade war escalation", "a rate surprise"],
    "product": ["AI chip", "quantum processor", "mixed-reality headset", "autonomous vehicle"],
    "feature": ["battery life", "processing speed", "neural efficiency"],
    "Tech": ["AI", "Blockchain", "EV", "5G", "Edge computing"],
    "achievement": ["outperforms all baselines", "reduces error by half"],
    "device": ["smartphone", "laptop", "wearable", "sensor array"],
    "metric": ["battery capacity", "compute efficiency", "display resolution"],
    "food": ["ultra-processed food", "red meat", "sugar", "plant protein"],
    "effect": ["increased inflammation", "lower cognitive scores", "heart risk"],
    "disease": ["cancer", "Alzheimer's", "Parkinson's", "diabetes"],
    "Treatment": ["Immunotherapy", "Gene therapy", "Mindfulness therapy", "Drug X"],
    "symptom": ["pain levels", "inflammation markers", "anxiety scores"],
    "technology": ["AI diagnostics", "telemedicine", "robotic surgery"],
    "outcome": ["recovery times", "misdiagnosis rates", "patient satisfaction"],
    "phenomenon": ["ice shelf collapse", "coral bleaching", "permafrost thaw"],
    "region": ["the Arctic", "Southeast Asia", "Sub-Saharan Africa"],
    "initiative": ["reforestation", "carbon tax", "ocean cleanup"],
    "pollutant": ["CO2", "methane", "particulate matter"],
    "trend": ["declines sharply", "recovers steadily", "faces extinction"],
    "Species": ["Polar bear", "Monarch butterfly", "Snow leopard", "Blue whale"],
    "Athlete": ["A world champion", "A rising star", "An Olympic veteran"],
    "event": ["the 100m sprint", "the marathon", "the high jump"],
    "time": ["9.84 seconds", "2:01:34", "an all-time best"],
    "team": ["home side", "underdog squad", "defending champions"],
    "score": ["3-2", "4-1", "2-0"],
    "League": ["The national league", "The professional circuit", "The governing body"],
    "number": ["three", "four", "five", "six"],
    "player": ["the star forward", "the top-ranked athlete", "the team captain"],
    "duration": ["two years", "six months", "the rest of the season"],
    "theory": ["quantum mechanics", "evolutionary biology", "cosmological models"],
    "material": ["superconductor", "metamaterial", "bio-composite"],
    "property": ["zero electrical resistance", "negative refractive index"],
    "structure": ["proteome", "neural connectome", "genome"],
    "organism": ["a deep-sea microbe", "a rare fern species"],
    "telescope": ["James Webb Space Telescope", "Event Horizon Telescope"],
    "celestial_object": ["a black hole", "an exoplanet atmosphere", "a neutron star merger"],
    "bill": ["climate legislation", "privacy reform", "digital rights"],
    "margin": ["a narrow two-vote", "a comfortable 60-40", "a historic supermajority"],
    "agreement": ["a climate accord", "a trade deal", "a ceasefire"],
    "summit": ["G7", "UN Climate Conference", "bilateral summit"],
    "action": ["early elections", "an independent inquiry", "resignation"],
    "controversy": ["corruption", "data leak", "espionage"],
    "party": ["the ruling party", "the opposition", "a coalition"],
    "score": ["94.7", "88.2", "a new state-of-the-art"],
    "benchmark": ["MMLU", "SuperGLUE", "HumanEval"],
    "technique": ["chain-of-thought prompting", "retrieval augmentation", "constitutional AI"],
    "capability": ["reasoning", "factual accuracy", "instruction following"],
    "model": ["a 70B parameter model", "a 7B fine-tuned model"],
    "data_size": ["1 trillion", "500 billion", "200 billion"],
    "achievement": ["diagnoses rare diseases", "writes production code"],
    "domain": ["medicine", "law", "scientific research"],
}

LABEL_SETS = {
    "Finance": ["Finance", "Economics", "Business", "Policy", "Markets"],
    "Technology": ["Technology", "Innovation", "Business", "AI", "Science"],
    "Health": ["Health", "Medicine", "Research", "Biology", "Society"],
    "Environment": ["Environment", "Climate", "Science", "Policy", "Sustainability"],
    "Sports": ["Sports", "Entertainment", "Achievement", "Business"],
    "Science": ["Science", "Research", "Physics", "Biology", "Technology"],
    "Politics": ["Politics", "Diplomacy", "Law", "Society", "Policy"],
    "AI": ["AI", "Technology", "Science", "Ethics", "Innovation"],
}


def fill_template(template: str) -> str:
    """Fill template placeholders with random fillers."""
    import re
    keys = re.findall(r"\{(\w+)\}", template)
    for key in keys:
        if key in FILLERS:
            template = template.replace(f"{{{key}}}", random.choice(FILLERS[key]), 1)
    return template


def generate_template_sample(category: str) -> Dict:
    templates = TEMPLATES[category]
    template = random.choice(templates)
    text = fill_template(template)
    labels = random.sample(LABEL_SETS[category], k=random.randint(2, 3))
    return {"text": text, "labels": labels}


def generate_template_data(num_samples: int) -> List[Dict]:
    categories = list(TEMPLATES.keys())
    data = []
    for _ in range(num_samples):
        cat = random.choice(categories)
        data.append(generate_template_sample(cat))
    return data


# ---------------------------------------------------------------------------
# OpenAI-based generation
# ---------------------------------------------------------------------------

OPENAI_SYSTEM_PROMPT = """You are a data generation assistant. Generate diverse, realistic sentences 
about various topics. Each sentence should be factual-sounding news or information."""

OPENAI_USER_PROMPT = """Generate {n} diverse text classification examples.
Each example must be a JSON object with:
- "text": a single realistic sentence (20-40 words) about a real-world topic
- "labels": a list of 2-3 relevant category labels

Use diverse topics: Finance, Technology, Health, Environment, Sports, Politics, Science, AI, 
Culture, Education, Law, Agriculture, Space, Energy, Food, etc.

Return ONLY a valid JSON array, no extra text."""


def generate_openai_data(num_samples: int, api_key: str, batch_size: int = 20) -> List[Dict]:
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    all_data = []

    for batch_start in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - batch_start)
        print(f"  Generating batch {batch_start // batch_size + 1} ({n} samples)...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                    {"role": "user", "content": OPENAI_USER_PROMPT.format(n=n)},
                ],
                temperature=0.9,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            batch = json.loads(raw)
            all_data.extend(batch)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"  ⚠️  Error in batch: {e}. Using template fallback for this batch.")
            for _ in range(n):
                cat = random.choice(list(TEMPLATES.keys()))
                all_data.append(generate_template_sample(cat))

    return all_data[:num_samples]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic classification data")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_data.json", help="Output file path")
    parser.add_argument("--no_api", action="store_true", help="Use template-based generation (no API)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"🔧 Generating {args.num_samples} samples...")

    if args.no_api:
        print("   Using template-based generation (no API)")
        data = generate_template_data(args.num_samples)
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  No OpenAI API key found. Falling back to template-based generation.")
            data = generate_template_data(args.num_samples)
        else:
            print("   Using OpenAI gpt-4o-mini")
            data = generate_openai_data(args.num_samples, api_key)

    # Merge with existing data if the file exists
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        data = existing + data
        print(f"   Merged with {len(existing)} existing samples → {len(data)} total")

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {len(data)} samples to {args.output}")


if __name__ == "__main__":
    main()
