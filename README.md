# Encoded-Inequality-Thesis
This repository supports the MA thesis "Encoded Inequality: Tracing Gender Bias from Wikipedia to ChatGPT through Computational and Discursive Analysis" (2025). It presents a hybrid methodology combining word embedding analysis, dependency parsing, and discourse theory to examine how gendered occupational bias circulates from Wikipedia (as a proxy training corpus) into ChatGPT's language outputs.


# Research Questions
RQ1: What patterns of gender-specific attributions can be identified in the work-related subset of the English-language Wikipedia dataset used to train OpenAI’s LLM GPT?
RQ2: How does ChatGPT reproduce or amplify gender-specific attributes in its outputs, particularly in work-related contexts, where gender-neutral responses are expected?
RQ3: To what extent do the gender-specific patterns identified in the Wikipedia subset align with the adjectives and roles assigned to men and women in ChatGPT’s outputs?


# Methodology Overview
Wikipedia Adjective Bias Pipeline
Corpus Construction: Wikipedia articles on professions collected via SPARQL (see download_wikidatapedia.txt)
Gender Attribution: Based on Wikidata gender markers (Q6581097 = male, Q6581072 = female)
Adjective Extraction:
find_bias_plain.py – baseline extractor
find_bias_adjectives_only.py – filters adjective-only bias
Word Embedding:
gendered_adjective_semantic_clustering_word2vec.py
similaradjectives_word2vec.py – semantic expansion
Dependency Parsing:
dependency_parsing_per_gender.txt – uses spaCy's DependencyMatcher 
Frequency & Sentiment:
gendered_adjective_analysis_frequency_sentiment.py – uses VADER for sentiment scoring
Network Construction:
network.txt – hyperlink-based gendered profession network using networkx
ChatGPT Output Analysis
Prompts around job roles, personality traits, negotiation, and leadership analyzed
Manual coding of adjective use across male and female targets
Temporal comparison of responses from ChatGPT (2024 vs 2025)


| Category            | Tools Used                         |
| ------------------- | ---------------------------------- |
| NLP & Parsing       | spaCy, NLTK, VADER                 |
| Word Embeddings     | Gensim (Word2Vec)                  |
| Web Data Collection | Pywikibot, requests, BeautifulSoup |
| Visualization       | NetworkX, Matplotlib               |
| Data Processing     | pandas, pickle, csv                |


# Reproducing the Analysis — Thesis-Based Pipeline
Step 1: Corpus Construction & Filtering
Run download_wikidatapedia.py to query Wikidata and collect gender-labeled Wikipedia biographies with known professions (P106/P39).
Use a Wikipedia dump to extract article content, creating a balanced corpus of ~696,000 articles (50/50 male-female).
Step 2: Lexicon Expansion via Word2Vec
Use similaradjectives_word2vec.py to expand a seed list of culturally gendered adjectives based on semantic similarity.
Word2Vec embeddings trained on Wikipedia ensure interpretability and historical relevance.
Step 3: Adjective Extraction (Dependency Parsing)
Use dependency_parsing_per_gender.py (based on spaCy en_core_web_lg) to parse all biographies.
Extract all adjectives (not limited to noun-adjective pairs), grouped by the subject's gender.
Step 4: Frequency Filtering
Run find_bias_adjectives_only.py to:
Filter adjectives based on a curated list (adjectives_list.txt)
Count each adjective once per article (to avoid repetition bias)
Output frequency tables by gender to bias_adjectives_only.csv
Step 5: Manual Validation
Use a structured 6-point rubric to manually filter for gender-relevant descriptors (e.g., emotional, behavioral, evaluative).
Exclude technical, filler, and nationality-based adjectives.
Final lists: ~300 adjectives per gender.
Step 6: Computational Analysis
Analyze and visualize using the following:
Log-ratio comparison for frequency differences.
Gender exclusivity bar plots.
Top 15 biased adjectives by ratio.
Sentiment scoring (VADER) and gender averages.
Word clouds for qualitative visualization.
Semantic clustering (Word2Vec + k-means) to reveal gendered thematic groupings.
