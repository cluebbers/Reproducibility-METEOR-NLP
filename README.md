# Meh-Tricks: Towards Reproducible Results in NLP

## Introduction

This repository is for the Modul M.Inf.2801 Research Lab Rotation at the Georg-August Universität Göttingen.

This work is based on [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023) and the corresponding [Rogue Scores: Reproducibility Guide](https://analyses.org/2023/rogue-scores/reproduce/). After evaluating over 2000 papers using the ROUGE score, they reported three conclusions:

- (A) Critical evaluation decisions and parameters are routinely omitted, making most reported scores irreproducible.
- (B) Differences in evaluation protocol are common, affect scores, and impact the comparability of results reported in many papers.
- (C) Thousands of papers use nonstandard evaluation packages with software defects that produce provably incorrect scores.

Our work primarily investigates the METEOR score, its implementations, and its use in scientific papers. We implement a semi-automated literature review and identify decisions and aspects responsible for discrepancies in the use of METEOR.

It also provides some review of BLEU scores and reproduces the paper review of ROUGE scores.

## Methodology

### Systematic Literature Review

METEOR is a parameterized metric. While this makes it flexible, it can be difficult to report correctly.
There also are 5 different official versions of METEOR, v1.0 to 1.5.
The metric evolved and gained new parameters that changed the score.
The parameters in the official package are tuned to specific datasets and specific tasks.
Any deviation in parameters or failure to report the used METEOR version can potentially hinder the reproducibility of the scoring.

#### Data Collection

We use the [ACL Anthology Dataset](https://huggingface.co/datasets/ACL-OCL/ACL-OCL-Corpus) containing citations of 73285 machine learning papers, most of them with full texts.

#### METEOR Identification

Initially, we searched the dataset for occurrences of the keyword `meteor` to filter relevant papers.

#### Paper Review

All found papers are searched with regular expressions for

- Parameters as a string of letters and numbers
- Protocol variants taken from the [official documentation](https://www.cs.cmu.edu/~alavie/METEOR/README.html)
- Evaluation packages that were identified through online searches and manual codebase reviews.

#### Code Review

We use regular expressions to search for links or mentions of code repositories (GitHub, GitLab, Bitbucket, Google Code, Sourceforge) within the papers.
We use the GitHub API to search all GitHub repositories for the keyword `meteor`.
The decision for GitHub repositories was made because

- this was done in the original ROUGE paper
- GitHub repositories are by far the most used ones (see Evaluation)

All found repositories are manually reviewed for reproducibility. Detailed criteria for code can be found in [Appendix B](https://analyses.org/2023/rogue-scores/files/ACL2023-RogueScores.pdf).

#### Defining Reproducibility

There are three possibilities for reproducibility, based on [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023):

1. The paper cites the METEOR package and parameters
2. The Paper cites METEOR Package which needs no configuration
3. The Codebase includes a complete METEOR evaluation.

### Software Validation Testing

The original METEOR score software is a Java package.
It cannot be used directly in the popular machine-learning language Python.
So there needs to be a wrapper or a reimplementation.
Those are sources of failure.
We check how those implementations may differ.

#### Package Collection

We download all METEOR packages available for Python3 with more than 3 citations, resulting in 6 total packages.
They are identified by the code review and online search.

#### Specimen Task

Identical to the ROUGE paper, we use the [CNN / Daily Mail dataset](https://huggingface.co/datasets/ccdv/cnn_dailymail).
Human-written bullet point "highlights" are reference summaries.
We use METEOR to evaluate the specimen model hypothesis against the references using a development set.
See Appendix E in [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023).

#### Specimen Model

Evaluation is performed on Lead-3. This summarizes an article by returning its first three sentences.
See Appendix F in [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023).

#### Experimental Setup

We calculate the METEOR score for the generated summaries using the packages.
We only checked packages compatible with Python3.
Older packages were not considered for recency and simplicity.
The CNN / Daily Mail development set consists of 13.000 test cases.
We calculate the mean METEOR score for them.
To run the packages, open the docker image and run

```{python}
generate_packages_table()
```

## Requirements

### Jupyter Notebooks

- You need to install the following Python packages for the Jupyter notebooks:

```{python}
pandas=2.1.4
numpy=1.26.2  
requests=2.31.0
datasets=2.16.1
```

- The ACL Anthology dataset by Rohatgi et al. is available [here](https://huggingface.co/datasets/ACL-OCL/ACL-OCL-Corpus). `acl-publication-info.74k.v2.parquet` has to be in `\data`.
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

### Are METEOR parameters listed?

The regular expression to search for parameters anywhere in the text:

```{python}
pattern = r"((?: -[a-z123](?: [a-z0-9.]{1,4})?){2,})"
```

The parameters are copied to a string in `paper_meteor_params`.
Only 9 papers with parameters are found.
Those parameters are not related to METEOR.
This suggests METEOR is mostly run with the default configuration.
If no one messes with the tuned parameters, this ought to be a good thing.
The METEOR documentation states:

>"For the majority of scoring scenarios, only the -l and -norm options should be used. These are the ideal settings for which language-specific parameters are tuned."

### Which evaluation decisions are referenced?

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

The found protocol terms are added in a list in `paper_meteor_protocol`.
METEOR protocol terms are mentioned, mostly ones related to the tasks:

>- rank: parameters tuned to human rankings from WMT09 and WMT10
>- adq: parameters tuned to adequacy scores from NIST Open MT 2009
>- hter: parameters tuned to HTER scores from GALE P2 and
>- li: language-independent parameters

### Which METEOR software is cited?

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

The found terms are added to a list in `paper_meteor_packages`.
The most used packages are variations of Pycocoeval and NLTK.

| Github | number of citations |
|---------|---------|
| coco-caption | 128 |
| NLTK | 48 |
|generation eval | 44|
|huggingface/evaluate | 37|
| fairseq | 29 |
| nlg_eval | 8 |

### Does the paper include the released code?

The regular expression to search for codebases:
Found URLs are added to a list in `code_meteor_url`.
227 unique GitHub repositories were found in the papers.

Found GitHub codebases are searched for the term `meteor`.
If the keyword is found, it sets `code_meteor_prelim` to `True`.
55 of them contained the keyword `meteor`.

### Does the evaluation code appear reproducible?

All 55 code repositories were manually reviewed for reproducibility and what METEOR packages are used.
33 were found to be reproducible regarding the METEOR score.
This sets their `code_meteor`variable to `True`.

### How do different packages score the same input model?

You get the following results:
| GitHub | METEOR score|
|--------| ------------|
| WebNLG/GenerationEval  | 0.217303 |
| salaniz/pycocoevalcap    | 0.218336 |
| bckim92/language-evaluation | 0.167222 |
| Maluuba/nlg-eval | 0.221483 |
| Yale-LILY/SummEval | 0.221483 |
| nltk/nltk | 0.382306 |
| huggingface/evaluate | 0.382306 |
| facebookresearch/vizseq | 0.332000 |

## Results

### Reproducibility

Out of the 1.613 papers

- 16 % are deemed reproducible
- 34 % cite METEOR software packages
- 16 % release code
- 2 % release code with METEOR evaluation
- 0 % list METEOR configuration parameters

### Correctness

There seem to be two groups of implementations and something is off between them.

#### Wrappers

Packages with a lower score use the original Java METEOR implementation, its paraphrase file, and the wrapper from [coco-caption](https://github.com/tylin/coco-caption/blob/3a9afb2682141a03e1cdc02b0df6770d2c884f6f/pycocoevalcap/meteor/meteor.py).
They are used in roughly 60 % of the papers.
The wrapper was developed in cooperation with an author of METEOR, so we consider it correct.
It was written for Python2 and it had to be changed a bit to work with the Python3 packages we investigated.
Different adaptions for Python3 may have led to slightly different scores.

#### Reimplementation

Packages with the higher score use the reimplementation of NLTK with wordnet paraphrases.
They are used in roughly 40 % of the papers.
There is a [closed issue](https://github.com/nltk/nltk/issues/2655) on GitHub regarding the higher scores with NLTK.
NLTK implements METEOR v1.0.
Implementing METEOR v1.5 is far more complex.
It introduces paraphrases as an additional matching scheme and also word weights.
Default parameters for METEOR v1.0 were tuned for adequacy and fluency, while v1.5 uses rank as its default task.
NLTK developers are aware of the higher scores, but since they do not have an open-source implementation of METEOR v1.5, they stick with what they have.

They also use slightly different parameter values for alpha, beta, and gamma than the original JAVA implementation.

This could all be negated by citing the correct METEOR version when reporting METEOR scores.
The NLTK documentation does not mention the METEOR version.
This information is hidden in the [Python code](https://github.com/nltk/nltk/blob/f2a92bd7e360e39a4439e4d97540fd68f2721451/nltk/translate/meteor_score.py).
So no one references the appropriate METEOR v1.0 paper.
They all reference the most recent paper (METEOR v1.5).

## Conclusion

Low reproducibility was expected, as there is no reason why papers using METEOR should be significantly better than papers using ROUGE.
The numbers are a bit lower, but this could be rooted in a missing manual review of the papers.

Correctness is a more problematic issue.
NLTK and its derivates are used in about 40 % of the papers citing METEOR packages.
Chances are, researchers use them if they fit in their environment.
Maybe they are already using them for a different task.

If you try to build on a paper, there are these possibilities:

1. the paper reports the used packages and you use the same
2. you choose another implementation, and it's from the same group
3. the paper uses a wrapper and you use a reimplementation. Your scores will be higher by 10 points.
4. the paper uses a reimplementation and you use a wrapper. Your scores will be lower by 10 points.

Chances are progress is hindered or at least obscured by false comparisons.

The only upside: METEOR is not used on leaderboards at Papers With Code.

## Limitations

All limitations stated in [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023) apply.
Additionally:

- Our [ACL Anthology Dataset](https://huggingface.co/datasets/ACL-OCL/ACL-OCL-Corpus) only includes papers up to September 2022
- No manual paper review to filter for METEOR scoring
- Software validation testing was only done for packages available for Python3

## Contributing

Feel free to fork this repository and improve it.

- do a manual review of all papers
- continue the investigation on METEOR and BLEU
- build one docker image for investigating METEOR, BLEU, and ROUGE

### ROUGE

Recreating the investigation for reproducibility with the same regular expressions as [Rogue Scores](https://aclanthology.org/2023.acl-long.107) (Grusky, ACL 2023).

- 25 list parameters
- 459 list protocol terms
- 1168 list variants
- 274 list packages
- 62 are deemed reproducible (excluding Code Review)

| Finding | Rogue Scores | our reproduction |
|---|---|---|
| overall citations | 110.689 | 73.285 |
| manually excluded | 887 | 0 |
| papers in review | 2.834 | 2.593 |
|reproducible | 20 % | 2 % |
| list parameters | 5 % | 1 % |
| cite software package | 35 % | 11 % |

Our relative numbers are significantly lower because we did no manual review of automatically identified ROUGE papers.
This would have reduced the number of papers significantly.
Also, our reproducible score is lower because we did not do a code review.

### BLEU

This is only the start of an investigation.

- 54 list parameters
- 3.825 list protocol terms
- 631 list variants
- 5.336 list packages

| Finding | BLEU | ROUGE | METEOR |
|---|---|---|---|
| papers in review | 9.122 | 2.593 | 1.613 |
| list parameters | 0.6 % | 1 % | 0 % |
| cite software package | 58 % | 11 % | 34 % |
