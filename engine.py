import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from flashtext import KeywordProcessor
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = str(text).casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[']", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_companies(val):
    if isinstance(val, list):
        return [clean_text(item) for item in val if str(item).strip()]
    return []


def parse_owners(value):
    nums = re.findall(r"\d+", str(value).replace(",", ""))
    if not nums:
        return 0
    vals = [int(n) for n in nums]
    return int(sum(vals) / len(vals))


def ensure_derived_columns(df):
    if "cleaned_developers" not in df.columns and "developers" in df.columns:
        df["cleaned_developers"] = df["developers"].map(process_companies)
    elif "cleaned_developers" not in df.columns:
        df["cleaned_developers"] = [[] for _ in range(len(df))]

    if "cleaned_publishers" not in df.columns and "publishers" in df.columns:
        df["cleaned_publishers"] = df["publishers"].map(process_companies)
    elif "cleaned_publishers" not in df.columns:
        df["cleaned_publishers"] = [[] for _ in range(len(df))]

    if "owners_mid" not in df.columns:
        if "estimated_owners" in df.columns:
            df["owners_mid"] = df["estimated_owners"].map(parse_owners)
        elif "total_reviews" in df.columns:
            df["owners_mid"] = df["total_reviews"] * 30
        else:
            df["owners_mid"] = 0

    if "same_company" not in df.columns:
        df["same_company"] = [
            bool(devs) and devs == pubs
            for devs, pubs in zip(
                df["cleaned_developers"], df["cleaned_publishers"]
            )
        ]

    if "release_date" in df.columns and not np.issubdtype(
        df["release_date"].dtype, np.datetime64
    ):
        df["release_date"] = pd.to_datetime(
            df["release_date"], errors="coerce"
        )

    return df

GENERIC_COMPANY_STOPWORDS = {
    "games",
    "game",
    "player",
    "players",
    "studio",
    "studios",
    "entertainment",
    "interactive",
    "software",
    "digital",
    "media",
    "team",
    "corp",
    "co",
    "inc",
    "ltd",
    "llc",
    "steam",
    "nature",
    "play",
    "online",
    "action",
    "rpg",
    "indie",
    "best",
    "free",
}


def build_company_lookup(df):
    display = {}
    names = set()
    for raw_col, clean_col in (
        ("developers", "cleaned_developers"),
        ("publishers", "cleaned_publishers"),
    ):
        if raw_col not in df.columns or clean_col not in df.columns:
            continue
        for raw_list, clean_list in zip(df[raw_col], df[clean_col]):
            if not isinstance(raw_list, list) or not isinstance(
                clean_list, list
            ):
                continue
            for raw, cleaned in zip(raw_list, clean_list):
                if (
                    cleaned
                    and len(cleaned) >= 3
                    and cleaned not in GENERIC_COMPANY_STOPWORDS
                ):
                    names.add(cleaned)
                    display.setdefault(cleaned, raw)
    return sorted(names, key=len, reverse=True), display

def load_artifacts():
    model = joblib.load("models/linear_svc_classifier.joblib")
    title_vectorizer = joblib.load("models/title_vectorizer.joblib")
    metadata_vectorizer = joblib.load("models/metadata_vectorizer.joblib")
    df = ensure_derived_columns(joblib.load("models/games_data.joblib"))

    title_matrix = load_npz("models/title_matrix.npz")
    metadata_matrix = load_npz("models/metadata_matrix.npz")

    try:
        aux_data = joblib.load("models/auxiliary_data.joblib")
    except Exception:
        aux_data = {}

    title_list = aux_data.get(
        "TITLE_LIST", df["cleaned_name"].dropna().unique().tolist()
    )
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for title in title_list:
        keyword_processor.add_keyword(title, "game_title")

    company_list = aux_data.get("COMPANY_LIST")
    company_display = aux_data.get("COMPANY_DISPLAY")
    if not company_list or not company_display:
        company_list, company_display = build_company_lookup(df)

    title_set = set(title_list)
    company_processor = KeywordProcessor(case_sensitive=False)
    for name in company_list:
        if name not in title_set and name not in GENERIC_COMPANY_STOPWORDS:
            company_processor.add_keyword(name, name)

    metadata_terms = aux_data.get("METADATA_TERMS")
    if not metadata_terms:
        all_terms = set()
        for col in ["cleaned_categories", "cleaned_genres", "cleaned_tags"]:
            if col in df.columns:
                for term_list in df[col].dropna():
                    if isinstance(term_list, list):
                        all_terms.update(term_list)
        if not all_terms and "combined_metadata" in df.columns:
            all_terms = {
                "single player",
                "multiplayer",
                "co op",
                "pvp",
                "full controller support",
                "controller",
                "steam achievements",
                "steam cloud",
                "steam trading cards",
                "remote play together",
                "family sharing",
                "split screen",
                "steam workshop",
                "mmo",
                "indie",
                "action",
                "adventure",
                "casual",
                "rpg",
                "strategy",
                "simulation",
            }
        metadata_terms = sorted(list(all_terms), key=len, reverse=True)

    return {
        "model": model,
        "title_vectorizer": title_vectorizer,
        "metadata_vectorizer": metadata_vectorizer,
        "title_matrix": title_matrix,
        "metadata_matrix": metadata_matrix,
        "df": df,
        "responses_lookup": aux_data.get("responses_lookup", {}),
        "METADATA_TERMS": metadata_terms,
        "keyword_processor": keyword_processor,
        "company_processor": company_processor,
        "company_display": company_display,
    }


def mask_entities(clean_input, artifacts):
    matches = artifacts["keyword_processor"].extract_keywords(
        clean_input, span_info=True
    )

    extracted_title = None
    masked_text = clean_input
    if matches:
        longest_match = max(matches, key=lambda m: m[2] - m[1])
        start, end = longest_match[1], longest_match[2]
        extracted_title = clean_input[start:end]
        masked_text = clean_input[:start] + "game_title" + clean_input[end:]

    extracted_companies = []
    company_processor = artifacts.get("company_processor")
    if company_processor:
        company_matches = company_processor.extract_keywords(
            masked_text, span_info=True
        )
        selected = []
        occupied = []
        for name, start, end in sorted(
            company_matches, key=lambda m: m[2] - m[1], reverse=True
        ):
            if any(
                start < o_end and end > o_start for o_start, o_end in occupied
            ):
                continue
            occupied.append((start, end))
            selected.append((name, start, end))

        for name, start, end in sorted(
            selected, key=lambda m: m[1], reverse=True
        ):
            window = masked_text[
                max(0, start - 48) : min(len(masked_text), end + 24)
            ]
            placeholder = (
                "publisher" if re.search(r"\bpublish", window) else "developer"
            )
            extracted_companies.insert(0, name)
            masked_text = masked_text[:start] + placeholder + masked_text[end:]

    return masked_text, extracted_title, extracted_companies


def get_intent(
    clean_input,
    artifacts,
    confidence_threshold=0.35,
    margin_threshold=0.08,
):
    model = artifacts["model"]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([clean_input])[0]
        sorted_idx = np.argsort(probabilities)[::-1]
        best_idx = sorted_idx[0]
        second_idx = sorted_idx[1] if len(sorted_idx) > 1 else best_idx

        confidence = probabilities[best_idx]
        margin = probabilities[best_idx] - probabilities[second_idx]

        if confidence < confidence_threshold or margin < margin_threshold:
            return None
        return model.classes_[best_idx]

    elif hasattr(model, "decision_function"):
        scores = model.decision_function([clean_input])
        if scores.ndim == 1:
            best_idx = 1 if scores[0] > 0 else 0
        else:
            best_idx = np.argmax(scores[0])
        return model.classes_[best_idx]

    return model.predict([clean_input])[0]

def find_by_title(parsed_query, artifacts, similarity_threshold=0.15):
    title = parsed_query.get("title")
    if title is None:
        return None

    exact_match = artifacts["df"][artifacts["df"]["cleaned_name"] == title]
    if not exact_match.empty:
        return exact_match.iloc[0]

    query_vector = artifacts["title_vectorizer"].transform([title])
    similarities = cosine_similarity(
        query_vector, artifacts["title_matrix"]
    ).flatten()
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
    min_reviews=30,
    min_quality=0.35,
    min_year=None,
    min_owners=None,
):
    df = artifacts["df"]
    mask = (df["total_reviews"] >= min_reviews) & (
        df["quality_score"] >= min_quality
    )

    if include_free:
        mask &= df["price"] == 0
    elif max_price is not None:
        mask &= df["price"] <= max_price

    if platform in ["windows", "mac", "linux"] and platform in df.columns:
        mask &= df[platform] == True

    if metadata:
        for term in metadata:
            mask &= df["combined_metadata"].str.contains(
                rf"\b{re.escape(term)}\b", case=False, na=False
            )

    if min_year is not None and "release_date" in df.columns:
        years = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        mask &= years > min_year

    if min_owners is not None and "owners_mid" in df.columns:
        mask &= df["owners_mid"] >= min_owners

    return mask


def filter_games(
    artifacts,
    metadata=None,
    max_price=None,
    platform=None,
    top_n=5,
    sort_by="quality_score",
    include_free=False,
    min_year=None,
    min_owners=None,
    ascending=False,
):
    mask = get_game_filter(
        artifacts=artifacts,
        metadata=metadata,
        max_price=max_price,
        platform=platform,
        include_free=include_free,
        min_year=min_year,
        min_owners=min_owners,
    )
    filtered_df = artifacts["df"].loc[mask]

    if filtered_df.empty:
        return pd.DataFrame()

    if sort_by not in filtered_df.columns:
        sort_by = "quality_score"

    return filtered_df.sort_values(
        by=[sort_by, "total_reviews"], ascending=[ascending, False]
    ).head(top_n)


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


def format_results(results, header):
    if results.empty:
        return "No games found based on your criteria."

    response = f"{header}"
    for _, row in results.iterrows():
        price = f"\\${row['price']:.2f}" if row["price"] > 0 else "Free"
        response += f"  \n• **{row['name']}** — Price: **{price}** — Rating: **{row['quality_score'] * 100:.1f}%** ({row['total_reviews']:,} reviews)"
    return response

def handle_developer_publisher(clean_input, parsed_query, artifacts):
    df = artifacts["df"]
    companies = parsed_query.get("companies", [])
    primary_company = companies[0] if companies else None

    # Compare owners between two developers
    if "compare" in clean_input and len(companies) >= 2:
        c1, c2 = companies[0], companies[1]
        sum1 = df[df["cleaned_developers"].apply(lambda x: c1 in x)][
            "owners_mid"
        ].sum()
        sum2 = df[df["cleaned_developers"].apply(lambda x: c2 in x)][
            "owners_mid"
        ].sum()
        return (
            f"**Comparison of Estimated Popularity / Reach:**  \n"
            f"• **{artifacts['company_display'].get(c1, c1.title())}**: ~{sum1:,} total player reach  \n"
            f"• **{artifacts['company_display'].get(c2, c2.title())}**: ~{sum2:,} total player reach"
        )

    if re.search(
        r"\b(same developer and publisher|self published|developer and publisher are the same)\b",
        clean_input,
    ):
        matches = (
            df[df["same_company"]]
            .sort_values(by="quality_score", ascending=False)
            .head(5)
        )
        return format_results(
            matches,
            "Top rated self-published games (same developer & publisher):",
        )

    if "most indie" in clean_input:
        is_pub = "publish" in clean_input
        col = "cleaned_publishers" if is_pub else "cleaned_developers"
        indie_df = df[
            df["combined_metadata"].str.contains(
                r"\bindie\b", case=False, na=False
            )
        ]
        top_counts = indie_df.explode(col)[col].value_counts().head(5)
        res = f"Top {'Publishers' if is_pub else 'Developers'} with the most Indie titles:  \n"
        for comp, count in top_counts.items():
            if comp:
                name = artifacts["company_display"].get(comp, str(comp).title())
                res += f"• **{name}**: {count} games  \n"
        return res

    if not primary_company:
        return "Which developer or publisher would you like to know about?"

    display_name = artifacts["company_display"].get(
        primary_company, primary_company.title()
    )
    is_pub_query = "publish" in clean_input
    matched = (
        df[df["cleaned_publishers"].apply(lambda x: primary_company in x)]
        if is_pub_query
        else df[df["cleaned_developers"].apply(lambda x: primary_company in x)]
    )

    if matched.empty:
        return f"I couldn't find any games by **{display_name}** in the database."

    if re.search(r"\b(average price|cost on average)\b", clean_input):
        return f"The average price of games published by **{display_name}** is **\\${matched['price'].mean():.2f}**."

    if re.search(
        r"\b(most popular|most estimated owners|most owners)\b", clean_input
    ):
        top_game = matched.sort_values(
            by=["owners_mid", "total_reviews"], ascending=False
        ).iloc[0]
        owners_str = top_game.get("estimated_owners", f"{top_game['total_reviews'] * 30:,}+")
        return f"The most popular game by **{display_name}** is **{top_game['name']}** with an estimated **{owners_str}** owners."

    if re.search(r"\b(first game|oldest game|first release)\b", clean_input):
        oldest = (
            matched.dropna(subset=["release_date"])
            .sort_values(by="release_date")
            .iloc[0]
        )
        rel_str = pd.to_datetime(oldest["release_date"]).strftime("%B %d, %Y")
        return f"The first recorded game by **{display_name}** is **{oldest['name']}**, released on **{rel_str}**."

    if "free" in clean_input:
        return format_results(
            matched[matched["price"] == 0].head(5),
            f"Free games by **{display_name}**:",
        )

    if parsed_query.get("max_price") is not None:
        return format_results(
            matched[matched["price"] <= parsed_query["max_price"]].head(5),
            f"Games by **{display_name}** under **\\${parsed_query['max_price']:.2f}**:",
        )

    return format_results(
        matched.sort_values(by="quality_score", ascending=False).head(5),
        f"Games associated with **{display_name}**:",
    )


def handle_categories(clean_input, parsed_query, artifacts):
    matched_features = [
        term
        for term in artifacts["METADATA_TERMS"]
        if re.search(rf"\b{re.escape(term)}\b", clean_input)
    ]

    min_owners = None
    if re.search(
        r"\b(1000000|1 000 000|1 million|high estimated owners|lot of owners)\b",
        clean_input,
    ):
        min_owners = 1000000

    min_year = None
    year_match = re.search(r"\b(after|since)\s*(\d{4})\b", clean_input)
    if year_match:
        min_year = int(year_match.group(2))
    elif "recent" in clean_input:
        min_year = 2022

    results = filter_games(
        artifacts,
        metadata=matched_features,
        max_price=parsed_query.get("max_price"),
        platform=parsed_query.get("platform"),
        top_n=parsed_query.get("top_n", 5),
        include_free=parsed_query.get("include_free", False),
        min_year=min_year,
        min_owners=min_owners,
    )

    if results.empty:
        return "No games found matching all those specific features or criteria."

    desc = (
        f"Top games matching **{', '.join(matched_features)}**"
        if matched_features
        else "Recommended games"
    )
    return format_results(results, f"{desc}:")


def handle_question(clean_input, parsed_query, artifacts):
    game = find_by_title(parsed_query, artifacts)
    if game is None:
        return "I couldn't identify the game you're asking about. Could you please specify the game title?"

    if re.search(
        r"\b(who is the developer|who developed|who published|developer and publisher|who made)\b",
        clean_input,
    ):
        devs = ", ".join(game.get("developers", [])) if isinstance(game.get("developers"), list) else "Unknown"
        pubs = ", ".join(game.get("publishers", [])) if isinstance(game.get("publishers"), list) else "Unknown"
        if "developer and publisher" in clean_input or (
            "who developed" in clean_input and "published" in clean_input
        ):
            return f"**{game['name']}** was developed by **{devs}** and published by **{pubs}**."
        elif "publish" in clean_input:
            return f"**{game['name']}** was published by **{pubs}**."
        return f"**{game['name']}** was developed by **{devs}**."

    if re.search(r"\b(release date|when was|released|launch)\b", clean_input):
        if pd.isna(game.get("release_date")):
            return f"The release date for **{game['name']}** is not listed."
        date_str = pd.to_datetime(game["release_date"]).strftime("%B %d, %Y")
        return f"**{game['name']}** was released on **{date_str}**."

    if re.search(
        r"\b(estimated owners|how many owners|how popular)\b", clean_input
    ):
        owners = game.get("estimated_owners", f"{game.get('total_reviews', 0) * 30:,}+")
        return f"**{game['name']}** has an estimated owner count of **{owners}**."

    if re.search(
        r"\b(single player|multiplayer|both|co op|single-player)\b", clean_input
    ):
        meta = str(game.get("combined_metadata", "")).lower()
        is_sp = any(
            k in meta for k in ["single-player", "singleplayer", "single player"]
        )
        is_mp = any(k in meta for k in ["multi-player", "multiplayer", "co-op", "pvp"])
        if is_sp and is_mp:
            return f"**{game['name']}** supports **both Single-Player and Multiplayer** modes."
        elif is_mp:
            return f"**{game['name']}** is a **Multiplayer** game."
        elif is_sp:
            return f"**{game['name']}** is a **Single-Player** game."
        return f"Mode info for **{game['name']}**: {game.get('genres', 'N/A')}."

    if re.search(r"\b(genre|genres|categories)\b", clean_input):
        genres = (
            ", ".join(game.get("genres", []))
            if isinstance(game.get("genres"), list)
            else "N/A"
        )
        return f"**{game['name']}** belongs to the following genre(s): **{genres}**."

    return handle_game_info(parsed_query, artifacts)


def handle_game_info(parsed_query, artifacts):
    game = find_by_title(parsed_query, artifacts)
    if game is None:
        return "I couldn't find that specific game in the database. Could you check the title and try again?"

    price = f"\\${game['price']:.2f}" if game["price"] > 0 else "Free"
    discount = f" ({game['discount']}% off)" if game.get("discount", 0) > 0 else ""
    developers = (
        ", ".join(game["developers"])
        if isinstance(game.get("developers"), list) and len(game["developers"]) > 0
        else "Unknown"
    )
    genres = (
        ", ".join(game["genres"])
        if isinstance(game.get("genres"), list) and len(game["genres"]) > 0
        else "N/A"
    )
    return (
        f"**{game['name']}**  \n"
        f"**Description**: {game.get('short_description', 'N/A')}  \n"
        f"**Price**: {price}{discount}  \n"
        f"**Developer(s)**: {developers}  \n"
        f"**Genre(s)**: {genres}  \n"
        f"**Rating Score**: {game['quality_score'] * 100:.1f}% ({game['total_reviews']:,} reviews)"
    )


def handle_price_inquiry(parsed_query, artifacts):
    game = find_by_title(parsed_query, artifacts)
    if game is None:
        return "I couldn't identify the game you're asking about. Which game's price would you like to check?"

    discount = f" ({game['discount']}% off)" if game.get("discount", 0) > 0 else ""
    if game["price"] > 0:
        return f"{game['name']} is currently priced at **\\${game['price']:.2f}**{discount}."
    return f"{game['name']} is currently **Free to Play!**"


def handle_platform_support(parsed_query, clean_input, artifacts):
    if not parsed_query.get("title") and not parsed_query.get("platform"):
        return handle_categories(clean_input, parsed_query, artifacts)

    if not parsed_query.get("title") and parsed_query.get("platform"):
        return handle_recommendation(parsed_query, artifacts)

    game = find_by_title(parsed_query, artifacts)
    if game is None:
        return "I couldn't identify the game you're asking about. Which game's platforms would you like to check?"

    platforms = [
        name
        for name, col in (
            ("**Windows**", "windows"),
            ("**macOS**", "mac"),
            ("**Linux**", "linux"),
        )
        if game.get(col, False)
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
            similar = find_similar_games(
                game,
                artifacts,
                metadata=parsed_query.get("metadata"),
                max_price=parsed_query.get("max_price"),
                platform=parsed_query.get("platform"),
                include_free=parsed_query.get("include_free"),
                top_n=parsed_query.get("top_n", 5),
            )
            if similar.empty:
                return "I couldn't find any games similar to that title."
            return format_results(similar, f"Games similar to {game['name']}:")

    games = filter_games(
        artifacts,
        metadata=parsed_query.get("metadata"),
        max_price=parsed_query.get("max_price"),
        platform=parsed_query.get("platform"),
        top_n=parsed_query.get("top_n", 5),
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


def parse_query(clean_input, artifacts, title=None, companies=None):
    include_free = "free" in clean_input
    result = {
        "title": title,
        "companies": companies or [],
        "metadata": [
            t
            for t in artifacts["METADATA_TERMS"]
            if re.search(rf"\b{re.escape(t)}\b", clean_input)
        ],
        "max_price": None,
        "top_n": 5,
        "platform": None,
        "include_free": include_free,
    }

    numbers = re.findall(r"\d+(?:\.\d+)?", clean_input)
    if numbers:
        result["max_price"] = float(numbers[0])

    if re.search(r"\b(linux)\b", clean_input):
        result["platform"] = "linux"
    elif re.search(r"\b(mac(?:os)?)\b", clean_input):
        result["platform"] = "mac"
    elif re.search(r"\b(windows)\b", clean_input):
        result["platform"] = "windows"

    return result


def generate_response(raw_input, artifacts):
    clean_input = clean_text(raw_input)
    masked_text, extracted_title, extracted_companies = mask_entities(
        clean_input, artifacts
    )

    if (
        re.search(
            r"\b(single\s*player|multi\s*player|single-player|multiplayer)\b",
            clean_input,
        )
        and extracted_title
    ):
        intent = "question"
    elif re.search(
        r"\b(release date|when was|released|launch date)\b", clean_input
    ):
        intent = "question"
    elif re.search(
        r"\b(who (is the developer|developed|published|made)|how many owners|estimated owners|what genre)\b",
        clean_input,
    ):
        intent = "question"

    elif re.search(
        r"\b(controller|family sharing|remote play|split.?screen|achievements|cloud saves|trading cards|workshop|mmo|co.?op|pvp)\b",
        clean_input,
    ):
        intent = "categories"

    elif re.search(
        r"\b(developed by|published by|games by|compare .* owners|same developer and publisher|self published|most indie titles|most action rpg)\b",
        clean_input,
    ) or (
        extracted_companies
        and re.search(
            r"\b(games|titles|list|show|by|from|average price|first game)\b",
            clean_input,
        )
    ):
        intent = "developer_publisher"

    elif re.search(
        r"\b(price|how much|cost|free to play|is .* free|on sale|under \d+)\b",
        clean_input,
    ):
        intent = "price_inquiry"

    elif re.search(r"\b(linux|mac|macos|windows)\b", clean_input) and re.search(
        r"\b(support|run on|playable|compatible)\b", clean_input
    ):
        intent = "platform_support"

    elif re.search(
        r"\b(recommend|suggest|what should i play|games like|top rated|best games)\b",
        clean_input,
    ):
        intent = "recommendation"

    else:
        intent = get_intent(masked_text, artifacts)

    if (
        intent in artifacts["responses_lookup"]
        and not extracted_title
        and not extracted_companies
    ):
        return np.random.choice(artifacts["responses_lookup"][intent])

    if intent is None:
        return "I'm sorry, I didn't quite catch that. Could you try rephrasing your request?"

    parsed_query = parse_query(
        clean_input,
        artifacts,
        title=extracted_title,
        companies=extracted_companies,
    )

    if intent == "developer_publisher":
        return handle_developer_publisher(clean_input, parsed_query, artifacts)
    elif intent == "categories":
        return handle_categories(clean_input, parsed_query, artifacts)
    elif intent == "question":
        return handle_question(clean_input, parsed_query, artifacts)
    elif intent == "game_info":
        return handle_game_info(parsed_query, artifacts)
    elif intent == "recommendation":
        return handle_recommendation(parsed_query, artifacts)
    elif intent == "platform_support":
        return handle_platform_support(parsed_query, clean_input, artifacts)
    elif intent == "price_inquiry":
        return handle_price_inquiry(parsed_query, artifacts)

    return "I'm sorry, I didn't quite catch that. Could you try rephrasing your request?"