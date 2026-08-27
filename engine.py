import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
from flashtext import KeywordProcessor
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


def load_artifacts():
    model = joblib.load("models/linear_svc_classifier.joblib")
    title_vectorizer = joblib.load("models/title_vectorizer.joblib")
    metadata_vectorizer = joblib.load("models/metadata_vectorizer.joblib")
    df = joblib.load("models/games_data.joblib")

    title_matrix = load_npz("models/title_matrix.npz")
    metadata_matrix = load_npz("models/metadata_matrix.npz")

    aux_data = joblib.load("models/auxiliary_data.joblib")

    keyword_processor = KeywordProcessor(case_sensitive=False)
    for title in aux_data["TITLE_LIST"]:
        keyword_processor.add_keyword(title, "game_title")

    return {
        "model": model,
        "title_vectorizer": title_vectorizer,
        "metadata_vectorizer": metadata_vectorizer,
        "title_matrix": title_matrix,
        "metadata_matrix": metadata_matrix,
        "df": df,
        "responses_lookup": aux_data["responses_lookup"],
        "METADATA_TERMS": aux_data["METADATA_TERMS"],
        "keyword_processor": keyword_processor,
    }


def clean_text(text):
    text = str(text).casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[']", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mask_entities(clean_input, artifacts):
    matches = artifacts["keyword_processor"].extract_keywords(
        clean_input, span_info=True
    )

    if not matches:
        return clean_input, None

    longest_match = max(matches, key=lambda m: m[2] - m[1])
    _, start, end = longest_match

    extracted_title = clean_input[start:end]
    masked_text = clean_input[:start] + "game_title" + clean_input[end:]

    return masked_text, extracted_title


def get_intent(
    clean_input,
    artifacts,
    confidence_threshold=0.40,
    margin_threshold=0.10,
    verbose=False,
):
    probabilities = artifacts["model"].predict_proba([clean_input])[0]
    sorted_idx = np.argsort(probabilities)[::-1]

    best_idx = sorted_idx[0]
    second_idx = sorted_idx[1]

    confidence = probabilities[best_idx]
    margin = probabilities[best_idx] - probabilities[second_idx]

    if verbose:
        print("\n--- Intent Probabilities ---")
        for idx in sorted_idx:
            intent_name = artifacts["model"].classes_[idx]
            prob = probabilities[idx]
            print(f"  {intent_name:<20}: {prob * 100:6.2f}%")
        print(
            f"  Top Confidence: {confidence * 100:.2f}% | Margin: {margin * 100:.2f}%"
        )
        print("-----------------------------\n")

    if confidence < confidence_threshold or margin < margin_threshold:
        return None

    return artifacts["model"].classes_[best_idx]


def extract_metadata(clean_input, artifacts):
    return [
        term
        for term in artifacts["METADATA_TERMS"]
        if re.search(rf"\b{re.escape(term)}\b", clean_input)
    ]


def find_by_title(parsed_query, artifacts, similarity_threshold=0.15):
    title = parsed_query["title"]
    if title is None:
        return None

    exact_match = artifacts["df"][artifacts["df"]["cleaned_name"] == title]
    if not exact_match.empty:
        return exact_match.iloc[0]

    query_vector = artifacts["title_vectorizer"].transform([title])
    similarities = cosine_similarity(query_vector, artifacts["title_matrix"]).flatten()
    best_idx = np.argmax(similarities)
    if similarities[best_idx] < similarity_threshold:
        return None
    return artifacts["df"].iloc[best_idx]


def get_game_filter(
    artifacts,
    metadata=None,
    max_price=None,
    platform=None,
    include_free=False,
    min_reviews=1000,
    min_quality=0.65,
):
    mask = (artifacts["df"]["total_reviews"] > min_reviews) & (
        artifacts["df"]["quality_score"] >= min_quality
    )

    if include_free:
        mask &= artifacts["df"]["price"] == 0
    elif max_price is not None:
        mask &= artifacts["df"]["price"] <= max_price

    if platform in ["windows", "mac", "linux"]:
        mask &= artifacts["df"][platform] == True

    if metadata:
        for term in metadata:
            mask &= artifacts["df"]["combined_metadata"].str.contains(
                term, case=False, na=False
            )

    return mask


def find_similar_games(
    game,
    artifacts,
    metadata=None,
    max_price=None,
    platform=None,
    include_free=False,
    top_n=5,
):
    game_idx = game.name
    query_vector = artifacts["metadata_matrix"][game_idx]
    similarities = cosine_similarity(
        query_vector, artifacts["metadata_matrix"]
    ).flatten()

    mask = get_game_filter(
        artifacts=artifacts,
        metadata=metadata,
        max_price=max_price,
        platform=platform,
        include_free=include_free,
    )

    valid_indices = np.flatnonzero(mask)
    valid_indices = valid_indices[valid_indices != game_idx]

    if len(valid_indices) == 0:
        return pd.DataFrame()

    candidates = similarities[valid_indices]

    n = min(top_n, len(valid_indices))
    top_candidates = np.argpartition(candidates, -n)[-n:]
    top_candidates = top_candidates[np.argsort(-candidates[top_candidates])]
    return artifacts["df"].iloc[valid_indices[top_candidates]]


def filter_games(
    artifacts,
    metadata=None,
    max_price=None,
    platform=None,
    top_n=5,
    sort_by="quality_score",
    include_free=False,
):
    mask = get_game_filter(
        artifacts=artifacts,
        metadata=metadata,
        max_price=max_price,
        platform=platform,
        include_free=include_free,
    )
    filtered_df = artifacts["df"].loc[mask]

    if filtered_df.empty:
        return pd.DataFrame()

    return filtered_df.sort_values(
        by=[sort_by, "recommendations"], ascending=False
    ).head(top_n)


def format_results(results, header):
    if results.empty:
        return "No games found based on your criteria."

    response = f"{header}"
    for _, row in results.iterrows():
        price = rf"\${row['price']:.2f}" if row["price"] > 0 else "Free"
        response += f"  \n• **{row['name']}** - Price: **{price}** - Rating: **{row['quality_score'] * 100:.1f}%** ({row['total_reviews']:,} reviews)"
    return response


def handle_game_info(parsed_query, artifacts):
    game = find_by_title(parsed_query, artifacts)
    if game is None:
        return "I couldn't find that specific game in the database. Could you check the title and try again?"

    price = f"${game['price']:.2f}" if game["price"] > 0 else "Free"
    discount = f" ({game['discount']}% off)" if game["discount"] > 0 else ""
    developers = (
        ", ".join(game["developers"]) if len(game["developers"]) > 0 else "Unknown"
    )
    genres = ", ".join(game["genres"]) if len(game["genres"]) > 0 else "N/A"
    return (
        f"**{game['name']}**  \n"
        f"**Description**: {game['short_description']}  \n"
        f"**Price**: {price} {discount}  \n"
        f"**Developer(s)**: {developers}  \n"
        f"**Genre(s)**: {genres}  \n"
        f"**Rating Score**: {game['quality_score'] * 100:.1f}% ({game['total_reviews']:,} reviews)"
    )


def handle_price_inquiry(parsed_query, artifacts):
    game = find_by_title(parsed_query, artifacts)
    if game is None:
        return "I couldn't identify the game you're asking about. Which game's price would you like to check?"

    discount = f" ({game['discount']}% off)" if game["discount"] > 0 else ""
    return f"{game['name']} is currently {f'priced at **${game['price']:.2f}**{discount}.' if game['price'] > 0 else '**Free to Play!**'}"


def handle_platform_support(parsed_query, artifacts):
    if not parsed_query.get("title") and parsed_query.get("platform"):
        return handle_recommendation(parsed_query, artifacts)

    game = find_by_title(parsed_query, artifacts)
    if game is None:
        return "I couldn't identify the game you're asking about. Which game's platforms would you like to check?"

    platforms = [
        name
        for name, column in (
            ("**Windows**", "windows"),
            ("**macOS**", "mac"),
            ("**Linux**", "linux"),
        )
        if game[column]
    ]
    return (
        f"{game['name']} is available on: {', '.join(platforms)}"
        if platforms
        else f"{game['name']} is **not** available on Windows, Mac, or Linux."
    )


def handle_recommendation(parsed_query, artifacts):
    if parsed_query.get("title"):
        game = find_by_title(parsed_query, artifacts)
        if game is not None:
            similar_games = find_similar_games(
                game,
                artifacts,
                metadata=parsed_query.get("metadata"),
                max_price=parsed_query.get("max_price"),
                platform=parsed_query.get("platform"),
                include_free=parsed_query.get("include_free"),
                top_n=parsed_query["top_n"],
            )
            if similar_games.empty:
                return "I couldn't find any games similar to that title."
            return format_results(similar_games, f"Games similar to {game['name']}:")

    games = filter_games(
        artifacts,
        metadata=parsed_query.get("metadata"),
        max_price=parsed_query.get("max_price"),
        platform=parsed_query.get("platform"),
        top_n=parsed_query.get("top_n"),
        include_free=parsed_query.get("include_free"),
    )

    header = ["Recommended games"]
    if parsed_query.get("platform"):
        header.append(f"on **{parsed_query['platform'].capitalize()}**")
    if parsed_query.get("max_price") is not None:
        header.append(rf"under **\${parsed_query['max_price']:.2f}**")
    if parsed_query.get("metadata"):
        header.append(f"matching **{', '.join(parsed_query['metadata'])}**")

    return format_results(games, f"{' '.join(header)}:")


def handle_response(parsed_query, intent, artifacts):
    if intent == "game_info":
        return handle_game_info(parsed_query, artifacts)
    elif intent == "recommendation":
        return handle_recommendation(parsed_query, artifacts)
    elif intent == "platform_support":
        return handle_platform_support(parsed_query, artifacts)
    elif intent == "price_inquiry":
        return handle_price_inquiry(parsed_query, artifacts)
    elif intent in artifacts["responses_lookup"]:
        return np.random.choice(artifacts["responses_lookup"][intent])


def parse_query(clean_input, artifacts, title=None):
    include_free = "free" in clean_input

    result = {
        "title": title,
        "metadata": extract_metadata(clean_input, artifacts),
        "max_price": None,
        "top_n": 5,
        "platform": None,
        "include_free": include_free,
    }

    numbers = re.findall(r"\d+(?:\.\d+)?", clean_input)
    if numbers:
        result["max_price"] = float(numbers[0])

    # Platform
    if re.search(r"\b(linux)\b", clean_input):
        result["platform"] = "linux"
    elif re.search(r"\b(mac(?:os)?)\b", clean_input):
        result["platform"] = "mac"
    elif re.search(r"\b(windows)\b", clean_input):
        result["platform"] = "windows"
    return result


def generate_response(raw_input, artifacts):
    clean_input = clean_text(raw_input)
    masked_text, extracted_title = mask_entities(clean_input, artifacts)
    intent = get_intent(masked_text, artifacts)

    if intent is None:
        return "I'm sorry, I didn't quite catch that. Could you try rephrasing your request?"

    parsed_query = (
        parse_query(masked_text, artifacts, title=extracted_title)
        if intent not in artifacts["responses_lookup"]
        else None
    )
    return handle_response(parsed_query, intent, artifacts)
