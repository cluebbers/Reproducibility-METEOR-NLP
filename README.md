# Meh-Tricks: Towards Reproducible Results in NLP

This repository is for the Modul M.Inf.2801 Research Lab Rotation at the Georg-August Universität Göttingen.

This work is based on [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023) and the corresponding [Rogue Scores: Reproducibility Guide](https://analyses.org/2023/rogue-scores/reproduce/). After evaluating over 2000 papers using the ROUGE score, they reported three conclusions:

- (A) Critical evaluation decisions and parameters are routinely omitted, making most reported scores irreproducible.
- (B) Differences in evaluation protocol are common, affect scores, and impact the comparability of results reported in many papers.
- (C) Thousands of papers use nonstandard evaluation packages with software defects that produce provably incorrect scores.

This work primarily investigates the METEOR score, its implementations, and its use in scientific papers. We implement a semi-automated literature review and identify decisions and aspects responsible for discrepancies in the use of METEOR.

It also provides some review of BLEU scores and reproduces the paper review of ROUGE scores.

## Methodology

### Systematic Literature Review

#### Data Collection

We use the [ACL Anthology Dataset](https://huggingface.co/datasets/ACL-OCL/ACL-OCL-Corpus) containing citations of 73285 machine learning papers, most of them with full texts.

#### METEOR Identification

Initially, search the dataset for occurrences of the keyword `meteor` to filter relevant papers.

#### Paper Review

All found papers are searched with regular expressions for

- Parameters as a string of letters and numbers
- Protocol variants taken from the [official documentation]
- Evaluation packages were identified through online searches and manual codebase reviews.

#### Code Review

We use regular expressions to search for links or mentions of code repositories GitHub, GitLab, Bitbucket, Google Code, Sourceforge) within the papers.
We use the GitHub API to search all GitHub repositories for the keyword `meteor`.
The decision for GitHub repositories was made because

- this was done in the original ROUGE paper
- GitHub repositories are by far the most used ones (see Evaluation)

All found repositories are manually reviewed. Detailed criteria for reproducible code can be found in [Appendix B](https://analyses.org/2023/rogue-scores/files/ACL2023-RogueScores.pdf).

#### Defining Reproducibility

There are three possibilities for reproducibility, based on [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023):

1. The paper cites the METEOR package and parameters
2. The Paper cites METEOR Package which needs no configuration
3. The Codebase includes a complete METEOR evaluation.

### Software Validation Testing

#### Package Collection

#### Specimen Task and Model

The cnn_dailymail dataset will serve as the basis for generating summaries using a simple Lead-3 model. This dataset is chosen for its wide acceptance in summarization tasks and the availability of gold standard summaries for evaluation.

Generate summaries for a subset of articles using the Lead-3 model to create a test corpus.

#### Experimental Setup

Calculate the METEOR score for the generated summaries using identified packages from the code review phase.

To check for implementation errors, we checked packages compatible with Python3. Older packages were not considered for actuality and simplicity.
To run the packages, open the docker image and run

```{python}
generate_packages_table()
```

Compare the METEOR scores across different packages, focusing on:

- Component Analysis: Break down the METEOR score into its constituent components to understand the variance across implementations.
- Package Evaluation: Identify any software defects or deviations from the standard METEOR calculation that could lead to inconsistent scores.

## Requirements

### Jupyter Notebooks

- You need to install the following Python packages for the Jupyter notebooks:

```{python}
pandas=2.1.4
numpy=1.26.2  
requests=2.31.0
datasets=2.16.1
```

- The ACL Anthology dataset by Rohatgi et al. is available [here](https://huggingface.co/datasets/ACL-OCL/ACL-OCL-Corpus). *acl-publication-info.74k.v2.parquet* has to be in \data.
- You need a [GitHub API Key](https://github.com/settings/tokens) to search the repository for keywords. Add it at the top of the Jupyter Notebooks.

### Docker image

For software validation testing and the creation of the graphics, there is a docker image similar to the one provided for ROUGE. The full guide for the source image is [here](https://analyses.org/2023/rogue-scores/reproduce/).

```{python}
# create the image. Download stuff
make build

# start image
make start

# clean after your done
make clean
```

When you are done, use this to

```{python}
# stop docker and return later to continue working
docker stop meteorscores

# clean everything. needs make start again
make clean
```

## Experiments

There are 73.285 papers in the dataset. 5.871 Papers without a full text get their `error_download` variable set to `True`.

Then, we identify papers containing the keyword `meteor` in the full text. This sets the variable `paper_meteor_prelim` of 1.613 papers to `True`. Only those papers are included in the following experiments.

There may be some false positive papers found by keyword search because it could also match papers focusing on astronomical analysis. It would need a manual paper review to exclude them.

### How many papers post Parameters

The regular expression to search for parameters anywhere in the text:

```{python}
pattern = r"((?: -[a-z123](?: [a-z0-9.]{1,4})?){2,})"
```

- Only 9 papers with parameters are found. And those parameters are not related to METEOR. This suggests METEOR is mostly run with the default configuration. The METEOR documentation states:

>"For the majority of scoring scenarios, only the -l and -norm options should be used. These are the ideal settings for which language-specific parameters are tuned."
If no one messes with the tuned parameters, this ought to be a good thing.

### Protocol

The regular expressions to search for protocol-related terms within 500 characters of "meteor":

```{python}
regex_meteor_protocol = {
    'rank': r'\b(?:rank|ranking|WMT09|WMT10)\b',
    'adq': r'\b(?:adq|adequacy|NIST Open MT 2009)\b',
    'hter': r'\b(?:hter|HUMAN-targeted translation edit rate|GALE P2|GALE P3)\b',
    'li': r'\b(?:li|language[- ]independent)\b',
    'tune': r'\b(?:tune|tuning|parameter optimization)\b',
    'modules': r'\b(?:-m\s+(?:exact|stem|synonym|paraphrase)|module|exact|stem|synonym|paraphrase)\b',
    'normalize': r'\b(?:-norm|normalize|normalization|tokenize\s+and\s+lowercase|normalize\s+punctuation)\b',
    'lowercase': r'\b(?:-lower|lowercase|lowercasing|casing)\b',
    'ignore_punctuation': r'\b(?:-noPunct|no\s+Punctuation|ignore\s+punctuation|remove\s+punctuation)\b',
    'character_based': r'\b(?:-ch|character[- ]based|calculate\s+character[- ]based\s+precision\s+and\s+recall)\b',
    'verbose_output': r'\b(?:-vOut|verbose\s+output|output\s+verbose\s+scores)\b'
}
```

- METEOR protocol terms are mentioned, especially ones related to the tasks:

>- rank: parameters tuned to human rankings from WMT09 and WMT10
>- adq: parameters tuned to adequacy scores from NIST Open MT 2009
>- hter: parameters tuned to HTER scores from GALE P2 and
>- li: language-independent parameters

It is good that they are mentioned because this could potentially alter the scores.

### Packages

The regular expressions to search for METEOR packages anywhere in the paper:

```{python}
regex_meteor_packages = {
    'Meteor_coco': r'(?:coco.*?meteor|meteor.*?coco)',
    'pymeteor': r'pymeteor|zembrodt',   
    'generationeval_meteor': r'generationeval.*?meteor|meteor.*?generationeval|webnlg.*?meteor|meteor.*?webnlg',
    'evaluatemetric_meteor': r'evaluatemetric.*?meteor|meteor.*?evaluate(?:-)?metric',    
    'fairseq_meteor': r'fairseq.*?meteor|meteor.*?fairseq',
    'nlgeval_meteor': r'nlgeval.*?meteor|meteor.*?nlg(?:-)?eval',
    'nltk_meteor': r'nltk.*?meteor|meteor.*?nltk',
    'meteorscorer': r'meteorscorer',
    'beer_meteor': r'beer.*?meteor|meteor.*?beer',
    'compare_mt_meteor': r'compare(?:-)?mt.*?meteor|meteor.*?compare(?:-)?mt',
    'pysimt_meteor': r'pysimt.*?meteor|meteor.*?pysimt',
    'blend_meteor': r'blend.*?meteor|meteor.*?blend',
    'stasis_meteor': r'stasis.*?meteor|meteor.*?stasis',
    'huggingface' : r'hugging\s*face.*?evaluate.*?meteor|meteor.*?hugging\s*face.*?evaluate',
    # Template for other versions
    'Meteor_x': r'Meteor\s+[xX]\.?[\d\.]*', 
}
```

- The most used packages are variations of Pycocoeval and NLTK

### Codebases

The regular expression to search for codebases:

```{python}
regex_codebases = r'https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|sourceforge\.net|google\.code|code\.google)[^\s)]*(?<!\.)'
```

Found GitHub codebases are searched for the term "meteor". Matches are manually reviewed for reproducibility and what METEOR packages are used.
227 unique GitHub repositories were found in the papers. 55 of them contained the keyword "meteor". 33 were found to be reproducible regarding the METEOR score.

The docker file

## Results

### Reproducibility

### Correctness

You get the following results:
|package | METEOR score|
|--------| ------------|
|Pycoco  ||
|NLTK    ||

There seem to be two groups of implementations and something is off between them.
Packages with a lower score use the original Java METEOR implementation, its paraphrase file, and the wrapper by Pycocoevalcap.
Packages with the higher score use the reimplementation of NLTK with wordnet paraphrases.

## Discussion

There is a [closed issue](https://github.com/nltk/nltk/issues/2655) on GitHub regarding the higher scores with NLTK. NLTK implements METEOR v1.0. They are aware of the higher scores, but since they do not have an open-source implementation of METEOR v1.5, they stick with what they have.

This is problematic. NLTK is a very popular package and is utilized by other popular packages like Fairseq and Huggingface.
If you try to build on a paper that used NLTK, but for whatever reason decide to use a Java wrapper, your scores will be lower just because of that.

No one references the appropriate METEOR v1.0 paper. They all reference the most recent paper (METEOR v1.5).

Also, METEOR v1.0 had adequacy and fluency as a default task, while METEOR v1.5 has rank, resulting in different parameters.

The good: it is not used on leaderboards at Paperswithcode.

## Limitations

All limitations stated in [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023) apply.
Additionally:

- Our [ACL Anthology Dataset](https://huggingface.co/datasets/ACL-OCL/ACL-OCL-Corpus) only includes papers up to September 2022
- No manual paper review to filter for METEOR scoring
- Software validation testing was only done for packages available for Python3

## ROUGE

## BLEU

## Contributing
