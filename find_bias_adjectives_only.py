import sqlite3
import os
import spacy
from collections import Counter
import argparse



def load_texts_by_gender(db_path='extracted_pages.db', per_gender=348263):
    """
    Returns a dict with keys 'f' and 'm', each mapping to a list of page texts for that gender.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    pages = {'f': [], 'm': []}
    # debug: print distinct gender values present
    cursor.execute("SELECT DISTINCT gender FROM pages where gender not like 'unknown'")
    distinct = [row[0] for row in cursor.fetchall()]
    print(f"Distinct gender values in DB: {distinct}")
    # map keys to stored DB gender codes
    # detect stored gender codes
    cursor.execute("SELECT DISTINCT gender FROM pages where gender not like 'unknown'")
    distinct = [row[0] for row in cursor.fetchall()]
    print(f"Distinct gender values in DB: {distinct}")
    # map keys to actual stored codes
    if 'female' in distinct and 'male' in distinct:
        gender_map = {'f': 'female', 'm': 'male'}
    elif 'f' in distinct and 'm' in distinct:
        gender_map = {'f': 'f', 'm': 'm'}
    else:
        raise ValueError(f"Unexpected gender codes in DB: {distinct}")
    for key, sql_gender in gender_map.items():
        # debug: count available pages per gender
        cursor.execute("SELECT COUNT(*) FROM pages WHERE gender=?", (sql_gender,))
        total = cursor.fetchone()[0]
        print(f"Found {total} pages with gender '{sql_gender}' in the database")
        cursor.execute(
            "SELECT text FROM pages WHERE gender=? ORDER BY RANDOM() LIMIT ?",
            (sql_gender, per_gender)
        )
        sampled = cursor.fetchall()
        pages[key] = [row[0] for row in sampled]
    conn.close()
    return pages


def load_adjectives(adj_path='adjectives_list.txt'):
    """Load lowercase lemmas from a newline-adjective list."""
    adjs = set()
    with open(adj_path, encoding='utf-8') as f:
        for line in f:
            val = line.strip().lower()
            if val and val not in adjs:
                adjs.add(val)
    return adjs


def compute_adj_freqs(texts, adj_set, nlp, n_process=1, batch_size=100):
    """Compute adjective lemma counts using spaCy pipe with multiprocessing. 
    Each adjective is counted only once per document."""
    freq = Counter()
    total_docs = len(texts)
    print(f"Processing {total_docs} documents...")
    
    # Use enumerate to track document count
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        # Print progress every 1000 documents
        if i > 0 and i % 1000 == 0:
            print(f"Processed {i}/{total_docs} documents ({i/total_docs*100:.1f}%)")
            
        # Track adjectives seen in this document
        doc_adjs = set()
        for token in doc:
            if token.pos_ == 'ADJ':
                lemma = token.lemma_.lower()
                if lemma in adj_set and lemma not in doc_adjs:
                    doc_adjs.add(lemma)
                    freq[lemma] += 1
    
    print(f"Finished processing {total_docs} documents.")
    return freq


def main():
    # parse optional args
    parser = argparse.ArgumentParser(description='Find adjective usage bias between genders')
    parser.add_argument('--biased-list', help='Path to file with a predefined adjective list')
    args = parser.parse_args()
    pages = load_texts_by_gender()
    print(f"Loaded {len(pages['f'])} female texts and {len(pages['m'])} male texts")
    # load adjectives and NLP
    if args.biased_list:
        adj_set = load_adjectives(adj_path=args.biased_list)
        print(f"Loaded predefined adjective list from {args.biased_list} ({len(adj_set)} entries)")
    else:
        adj_set = load_adjectives()
        print(f"Loaded {len(adj_set)} adjectives to track.")
    # load spaCy model with only tagger and lemmatizer
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    # allow larger docs and use multiple processes
    nlp.max_length = 2000000
    n_process = max(1, os.cpu_count() - 5)
    print(f"Using spaCy with n_process={n_process}, batch_size=200")

    # compute adjective frequencies using spaCy
    print("Computing adjective frequencies via spaCy...")
    freq_f = compute_adj_freqs(pages['f'], adj_set, nlp, n_process=n_process)
    freq_m = compute_adj_freqs(pages['m'], adj_set, nlp, n_process=n_process)
    # top-20 adjectives by frequency
    top_f = freq_f.most_common(500)
    top_m = freq_m.most_common(500)
    print("\nTop adjectives by frequency for female pages:")
    for adj, count in top_f:
        print(f"{adj}: {count}")
    print("\nTop adjectives by frequency for male pages:")
    for adj, count in top_m:
        print(f"{adj}: {count}")
    # write results to CSV
    import csv
    with open('bias_adjectives_only.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['female_adjective', 'female_count', 'male_adjective', 'male_count'])
        for (f_adj, f_count), (m_adj, m_count) in zip(top_f, top_m):
            writer.writerow([f_adj, f_count, m_adj, m_count])
    print("Results written to bias_adjectives_only.csv")
    
if __name__ == '__main__':
    main()