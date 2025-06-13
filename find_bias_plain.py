import sqlite3
import os
from collections import Counter
import re
import argparse
import sys, os
# add local cTFIDF implementation to path
cTFIDF_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cTFIDF')
if cTFIDF_path not in sys.path:
    sys.path.insert(0, cTFIDF_path)


def load_texts_by_gender(db_path='extracted_pages.db', per_gender=348263):#348263
    """
    Returns a dict with keys 'f' and 'm', each mapping to a list of page texts for that gender.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    pages = {'f': [], 'm': []}
    # debug: print distinct gender values present
    cursor.execute("SELECT DISTINCT gender FROM pages")
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
    """Load lowercase lemmas from a newline-adjective list (default: adjectives.txt)."""
    adjs = set()
    with open(adj_path, encoding='utf-8') as f:
        for line in f:
            val = line.strip().lower()
            if val:
                adjs.add(val)
    return adjs


def compute_adj_freqs(texts, adj_set, nlp, n_process=1, batch_size=100):
    """Count adjectives, with each adjective counted only once per document."""
    # simple regex-based adjective count (not using spaCy)
    freq = Counter()
    for text in texts:
        # Track adjectives seen in this document
        doc_adjs = set()
        # lowercase and tokenize words
        for word in re.findall(r"\w+", text.lower()):
            if word in adj_set and word not in doc_adjs:
                doc_adjs.add(word)
                freq[word] += 1
    return freq


def main():
    # parse optional args
    parser = argparse.ArgumentParser(description='Find adjective usage bias between genders')
    parser.add_argument('--biased-list', help='Path to file with a predefined adjective list')
    args = parser.parse_args()
    pages = load_texts_by_gender()
    print(f"Loaded {len(pages['f'])} female texts and {len(pages['m'])} male texts")
    # load adjectives
    if args.biased_list:
        adj_set = load_adjectives(adj_path=args.biased_list)
        print(f"Loaded predefined adjective list from {args.biased_list} ({len(adj_set)} entries)")
    else:
        adj_set = load_adjectives()
        print(f"Loaded {len(adj_set)} adjectives to track.")
      # compute class-level c-TF-IDF for female vs male corpora
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from cTFIDF import CTFIDFVectorizer
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please `pip install scikit-learn` and ensure cTFIDF dependencies (numpy, scipy, pandas) are installed to use local cTFIDF implementation")
        return
    corpora = [" ".join(pages['f']), " ".join(pages['m'])]
    # Count occurrences using vocabulary
    cnt_vect = CountVectorizer(vocabulary=adj_set)
    X_counts = cnt_vect.fit_transform(corpora)
    # apply c-TF-IDF
    # total number of documents used for IDF calculation
    total_docs = len(pages['f']) + len(pages['m'])
    ctfidf = CTFIDFVectorizer().fit(X_counts, n_samples=total_docs).transform(X_counts)
    features = cnt_vect.get_feature_names_out()
    # row0=female, row1=male
    scores_f = ctfidf[0].toarray().ravel() if hasattr(ctfidf[0], 'toarray') else ctfidf[0]
    scores_m = ctfidf[1].toarray().ravel() if hasattr(ctfidf[1], 'toarray') else ctfidf[1]
    # top‐20 per class
    top_f = sorted(zip(features, scores_f), key=lambda x: x[1], reverse=True)[:200]
    top_m = sorted(zip(features, scores_m), key=lambda x: x[1], reverse=True)[:200]
    #print("\nTop adjectives by c-TF-IDF for female pages:")
    #for adj, score in top_f:
    #    print(f"{adj}: {score:.4f}")
    #print("\nTop adjectives by c-TF-IDF for male pages:")
   # for adj, score in top_m:
   #     print(f"{adj}: {score:.4f}")
    # write results to CSV
    import csv
    with open('bias_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['male_adjective', 'male_score', 'female_adjective', 'female_score'])
        for (f_adj, f_score), (m_adj, m_score) in zip(top_f, top_m):
            writer.writerow([m_adj, m_score, f_adj, f_score])
    print("Results written to bias_results.csv")
     
if __name__ == '__main__':
    main()