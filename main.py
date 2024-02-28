from __future__ import annotations
import typing as T
import pandas as pd
import nltk
import logging; logging.disable(logging.CRITICAL)

# NLTK need wordnet for METEOR scoring
nltk.download('wordnet')

def collect(*patterns: str) -> list[T.Callable]:

    """

    Search this file for functions using regular expressions. For example, this
    function can be used to select a subset of experiments to run together.

    Example:

        >>> collect("^run_models")

    """

    import re
    import inspect

    match = []
    module = __import__(__name__)

    for k, v in list(globals().items()):

        if inspect.getmodule(v) != module:
            continue

        if not inspect.isfunction(v):
            continue

        if not any(re.search(p, k) for p in patterns):
            continue

        match.append(v)

    return list(match)

def cache(

        directory: str = "data/cache",
        save: T.Optional[T.Callable] = None,
        load: T.Optional[T.Callable] = None,
        suffix: T.Optional[str] = None,

        ) -> T.Callable:

    """

    Decorator that caches the (arbitrary type) return value of a no-argument
    function to disk using pickle/dill.

    """

    import pickle
    import functools
    import inspect
    import dill

    from datetime import datetime
    from pathlib import Path
    from tqdm import tqdm

    directory = Path(directory)
    directory.mkdir(parents = True, exist_ok = True)

    if save is None:
        save = lambda path, data: path.write_bytes(dill.dumps(data))
    if load is None:
        load = lambda path: dill.loads(path.read_bytes())
    if suffix is None:
        suffix = ""

    def decorator(fn: T.Callable) -> T.Callable:

        @functools.wraps(fn)
        def wrapper() -> T.Any:

            path = directory / f"{fn.__name__}{suffix}"

            start = datetime.now()

            if path.exists():

                result = load(path)
                finish = datetime.now()

                tqdm.write(f"[CACHED] {fn.__name__} ({finish - start})")

            else:

                tqdm.write(f"[RUNNING] {fn.__name__}")

                result = fn()
                save(path, result)
                finish = datetime.now()

                tqdm.write(f"[SAVED] {fn.__name__} ({finish - start})")

            return result

        return wrapper

    return decorator

def dfcache(directory: str = "data/cache") -> T.Callable:

    """

    Decorator that caches the Pandas DataFrame return value of a no-argument
    function to disk as a Parquet file.

    """

    import pandas as pd

    return cache(
        directory = directory,
        save = lambda path, data: data.to_parquet(path),
        load = lambda path: pd.read_parquet(path),
        suffix = ".parquet"
    )

def patch(patch_dict: dict[str, str]) -> T.Callable:

    """

    Patch the Python import system to temporarily install a package.

    We need this function because Python cannot natively install two packages
    with the same name. Unfortunately for us, basically half of these packages
    are called either "rouge" or "pyrouge."

    Example:

    >>> @patch({
    >>>     "pyrouge":
    >>>     "data/packages/andersjo_pyrouge/andersjo-pyrouge-3b6c415/pyrouge/__init__.py",
    >>> })

    """

    import functools

    def decorator(fn: T.Callable) -> T.Callable:

        @functools.wraps(fn)
        def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:

            from pathlib import Path

            import importlib
            import importlib.util
            from IPython.lib.deepreload import reload
            import uuid

            import sys
            old = sys.modules.copy()

            try:

                imports = {}

                for name, path_pattern in patch_dict.items():

                    path_matches = list(Path(".").glob(path_pattern))
                    assert len(path_matches) == 1, \
                        f"@patch path {path_pattern} matches too many paths."

                    path = str(path_matches[0])

                    spec = importlib.util.spec_from_file_location(name, path)
                    module = importlib.util.module_from_spec(spec)

                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    imports[name] = module

                delta = {
                    k:v for k,v in sys.modules.items()
                    if v not in old.values()
                }

                prefix = str(uuid.uuid1())

                result = fn(imports, *args, **kwargs)

                for k, v in delta.items():
                    sys.modules[prefix + k] = sys.modules[k]
                    del sys.modules[k]

            except Exception as e:

                sys.modules = old
                raise e

            sys.modules = old

            return result

        return wrapper

    return decorator

def dict_avg(list_of_dicts: list[dict]) -> dict:

    """

    Average a list of dicts into a single dict.

    Example:

    >>> _dict_avg([{"a": 1, "b": 2}, {"a":2, "b": 3}])
    {'a':1.5, 'b': 2.5}

    """

    total = {}

    for d in list_of_dicts:
        for k, v in d.items():
            if k not in total:
                total[k] = 0
            total[k] += v

    return {
        k: v / len(list_of_dicts)
        for k, v in total.items()
    }

def pairwise(

        pack: T.Callable,
        refs: list[str], hyps: list[str],
        *args: T.Any, **kwargs: T.Any

        ) -> list[dict[str, float]]:

    """

    Repeatedly run ROUGE on single pairs of references and hypotheses, returning
    a set of ROUGE scores for each reference-hypothesis pair.

    """

    import itertools
    import concurrent.futures as cf
    from tqdm import tqdm

    def batch(iterable: T.Iterable, batch_size: int) -> T.Iterable:

        import itertools
        iterator = iter(iterable)

        while batch := list(itertools.islice(iterator, batch_size)):
            yield batch

    results = []
    batches = list(zip(batch(refs, 100), batch(hyps, 100)))

    for refs_batch, hyps_batch in tqdm(batches, desc = "Pairwise"):

        with cf.ProcessPoolExecutor() as ex:

            futures = [
                ex.submit(pack, [ref], [hyp], *args, **kwargs)
                for ref, hyp in zip(refs_batch, hyps_batch)
            ]

            for _ in cf.as_completed(futures):
                pass

            results.extend([f.result() for f in futures])

    return results

@cache()

def meteor_data_dev() -> tuple[list[str], list[str]]:

    """

    Return (reference list, hypothesis list) tuple for CNN/DM dev split.

    """

    return meteor_data_references_hypotheses(split = "validation")

@cache()

def meteor_data_test() -> tuple[list[str], list[str]]:

    """

    Return (reference list, hypothesis list) tuple for CNN/DM test split.

    """

    return meteor_data_references_hypotheses(split = "test")

def meteor_data_references_hypotheses(

        split: str = "validation",
        count: T.Optional[int] = None,

    ) -> tuple[list[str], list[str]]:

    """

    Setup a split of the CNN/DM dataset and run the Lead-3 baseline on it.
    Return a tuple of two lists:

    (list of reference summaries, list of hypothesis summaries)

    """

    import datasets
    from tqdm import tqdm

    assert split in {"train", "validation", "test"}

    data = datasets.load_dataset(
        "cnn_dailymail", "3.0.0",
        split = split
    )

    if count:

        data = data.select(range(count))

    refs = data["highlights"]

    hyps = [
        meteor_data_lead3_baseline(s)
        for s in tqdm(
            data["article"],
            desc = f"Generating Hypotheses ({split})"
        )
    ]

    return refs, hyps

def meteor_data_lead3_baseline(article: str) -> str:

    """

    Run the Lead-3 baseline on an article text.

    """

    import nltk

    return "\n".join(nltk.sent_tokenize(article)[:3])

def meteor_install_path() -> str:

    """

    Return the directory containing the reference METEOR java script.

    """

    import os.path
    return os.path.abspath("data/libraries/")

def meteor_install_packages() -> None:

    """

    Download and install all ROUGE packages.

    """

    meteor_install_packages_file(
        path = "data/packages/nlg_eval",
        urls = [
            "https://raw.githubusercontent.com/Maluuba/nlg-eval/master/nlgeval/pycocoevalcap/meteor/meteor-1.5.jar",
            "https://raw.githubusercontent.com/Maluuba/nlg-eval/master/nlgeval/pycocoevalcap/meteor/meteor.py"
        ]
    )
    
    meteor_install_packages_file(
        path = "data/packages/nlg_eval/data",
        urls = [
            "https://raw.githubusercontent.com/Maluuba/nlg-eval/master/nlgeval/pycocoevalcap/meteor/data/paraphrase-en.gz"
        ]
    )
    
    meteor_install_packages_pypi(
        path = "data/packages/salaniz_pycocoevalcap",
        package = "pycocoevalcap",
        version = "1.2",
    )
    
    meteor_install_packages_pypi(
        path = "data/packages/yale_summeval",
        package = "summ_eval",
        version = "0.892",
    )
    
    meteor_install_packages_file(
        path = "data/packages/yale_summeval/summ_eval-0.892/summ_eval",
        urls = [
            "https://raw.githubusercontent.com/Maluuba/nlg-eval/master/nlgeval/pycocoevalcap/meteor/meteor-1.5.jar",
        ]
    )
    
    meteor_install_packages_file(
        path = "data/packages/yale_summeval/summ_eval-0.892/summ_eval/data",
        urls = [
            "https://raw.githubusercontent.com/Maluuba/nlg-eval/master/nlgeval/pycocoevalcap/meteor/data/paraphrase-en.gz"
        ]
    )

def meteor_install_packages_file(path: str, urls: list[str]) -> None:
    """
    Download a list of files from URLs.
    """
    from pathlib import Path
    import requests
    from tqdm import tqdm

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    for url in urls:
        file_name = url.split("/")[-1]
        file_path = path / file_name

        if file_path.exists():
            tqdm.write(f"[EXISTS] {file_path}")
            continue  # Skip to the next file if this one already exists

        tqdm.write(f"[DOWNLOADING] {file_path}")
        # Download and save the file
        response = requests.get(url)
        file_path.write_bytes(response.content)


def meteor_install_packages_tar(path: str, url: str) -> None:

    """

    Download and extract package from arbitrary .tar.gz URL.

    """

    from pathlib import Path
    import requests
    import tarfile
    from tqdm import tqdm

    path = Path(path)

    if path.exists():

        tqdm.write(f"[EXISTS] {path}")
        return

    path.mkdir(parents = True, exist_ok = True)

    tqdm.write(f"[INSTALLING] {path}")

    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj = response.raw, mode = "r|gz")
    file.extractall(str(path))

def meteor_install_packages_zip(path: str, url: str) -> None:

    """

    Download and extract package from arbitrary .zip URL.

    """

    from pathlib import Path
    import requests
    from zipfile import ZipFile
    from tqdm import tqdm
    import io

    path = Path(path)

    if path.exists():

        tqdm.write(f"[EXISTS] {path}")
        return

    path.mkdir(parents = True, exist_ok = True)

    tqdm.write(f"[INSTALLING] {path}")

    response = requests.get(url, stream=True).content
    file = ZipFile(io.BytesIO(response))
    file.extractall(str(path))

def meteor_install_packages_github(

        path: str,
        user: str,
        repo: str,
        ref: str,

        ) -> None:

    """

    Download and extract .tar.gz repository from GitHub.

    """

    meteor_install_packages_tar(
        path = path,
        url = f"https://github.com/{user}/{repo}/tarball/{ref}",
    )

def meteor_install_packages_pypi(

        path: str,
        package: str,
        version: str,

        ) -> None:

    """

    Download and extract a PyPI Python package.

    """

    base = "https://files.pythonhosted.org/packages/source"

    meteor_install_packages_tar(
        path = path,
        url = f"{base}/{package[:1]}/{package}/{package}-{version}.tar.gz",
    )

def meteor_install_packages_patch(

        paths: list[str],
        patches: dict[str, str],

        ) -> None:

    """

    Directly patches source code. While it's preferable to run packages exactly
    as specified, it's sometimes necessary to make small changes to packages to
    make them usable in this multi-ROUGE evaluation universe.

    """

    import re
    from pathlib import Path

    matches = {
        file_path
        for path_pattern in paths
        for file_path in Path(".").glob(path_pattern)
        if file_path.is_file()
    }

    for path in matches:

        original = path.read_text()
        text = original

        for pattern, match in patches.items():

            text = re.sub(
                pattern, match, text,
                flags = re.MULTILINE,
            )

        path.write_text(text)

        if original != text:
            print(f"PATCHED: {path}")

def rouge_reference_base(

        refs: list[str],
        hyps: list[str],
        args: str = "-n 2",

        ) -> str:

    """

    Run the reference Perl ROUGE script, returning its exact outputs string.

    Other functions (rouge_reference_individual and rouge_reference_overall) use
    regular expressions to search this output string and return scores. This
    function should be be run directly.

    """

    import tempfile
    import subprocess
    from pathlib import Path

    rouge_path = meteor_install_path()

    temp = tempfile.TemporaryDirectory()

    ref_dir = Path(temp.name) / "ref"
    hyp_dir = Path(temp.name) / "hyp"

    ref_dir.mkdir(parents = True, exist_ok = True)
    hyp_dir.mkdir(parents = True, exist_ok = True)

    conf_file = Path(temp.name) / "config.xml"

    conf_entries = []

    for i, (ref, hyp) in enumerate(zip(refs, hyps)):

        ref_path = ref_dir / f"{i}.spl"
        hyp_path = hyp_dir / f"{i}.spl"

        ref_path.write_text(ref)
        hyp_path.write_text(hyp)

        conf_entries.append(f"""
            <EVAL ID="{i}">
                <PEER-ROOT>{hyp_dir}</PEER-ROOT>
                <MODEL-ROOT>{ref_dir}</MODEL-ROOT>
                <INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>
                <PEERS>
                    <P ID="1">{i}.spl</P>
                </PEERS>
                <MODELS>
                    <M ID="1">{i}.spl</M>
                </MODELS>
            </EVAL>
        """)

    conf_file.write_text(f"""
        <ROUGE-EVAL version="1.55">
        {"".join(conf_entries)}
        </ROUGE-EVAL>
    """)

    output = subprocess.check_output([
        f"{rouge_path}/ROUGE-1.5.5.pl",
        "-e", f"{rouge_path}/data",
        *args.split(),
        "-a", str(conf_file)
    ]).decode()

    temp.cleanup()
    return output

def rouge_reference_individual(

        refs: list[str],
        hyps: list[str],
        args: str = "-n 2",

        ) -> list[dict[str, float]]:

    """

    Run official ROUGE, returning scores for each individual model output.
    This will NOT generate an overall ROUGE score, ONLY individual scores.

    """

    import re

    if "-d" not in args:
        args += " -d"

    output = rouge_reference_base(refs, hyps, args)

    eval_pat = r"(ROUGE-.*) Eval (.*)\.\d+ R:(.*) P:(.*) F:(.*)"
    evals: list[dict] = [{} for _ in range(len(refs))]

    for rtype, i, r, p, f in re.findall(eval_pat, output):

        rtype = rtype.lower()
        i = int(i)

        evals[i][f"{rtype}-r"] = float(r)
        evals[i][f"{rtype}-p"] = float(p)
        evals[i][f"{rtype}-f"] = float(f)

    return evals

def rouge_reference_overall(

        refs: list[str],
        hyps: list[str],
        args: str = "-n 2",

        ) -> dict[str, float]:

    """

    Run official ROUGE, returning overall ROUGE scores for the entire dataset.
    This will NOT generate individual scores for each model output.

    """

    import re

    output = rouge_reference_base(refs, hyps, args)

    pat = r"(ROUGE-.*?) Average_(.): (.*?) \(95\%-conf\.int\. (.*?) - (.*?)\)"
    scores = {}

    for rtype, measure, score, cb, ce in re.findall(pat, output):

        rtype = rtype.lower()
        measure = measure.lower()

        scores[f"{rtype}-{measure}"] = float(score)
        scores[f"{rtype}-{measure}_cb"] = float(cb)
        scores[f"{rtype}-{measure}_ce"] = float(ce)

    return scores

@patch({
    "meteor":
    "data/packages/nlg_eval/meteor.py",
})

def meteor_package_nlg_eval(env, refs, hyps):
    # Prepare the data for scoring by converting lists to dictionaries
    gts = {i: [ref] for i, ref in enumerate(refs)}
    res = {i: [hyp] for i, hyp in enumerate(hyps)}

    # The Meteor class is accessed from the env dictionary provided by the @patch decorator
    meteor_scorer = env['meteor'].Meteor()
    scores = []
    
    # Compute the METEOR score for each pair of reference and hypothesis
    for i in range(len(refs)):
        score, _ = meteor_scorer.compute_score({i: gts[i]}, {i: res[i]})  # Adjusted to use dictionaries directly
        scores.append(score)

    # Aggregate the scores into a single dictionary
    aggregated_scores = {
        "meteor_mean_score": sum(scores) / len(scores) if scores else 0
    }

    return aggregated_scores

@patch({
    "summ_eval.metric": "data/packages/yale_summeval/summ_eval-0.892/summ_eval/metric.py",
    "meteor_metric": "data/packages/yale_summeval/summ_eval-0.892/summ_eval/meteor_metric.py",
})

def meteor_package_yale_summeval(env, refs, hyps):
    # Access the MeteorMetric class through the env dictionary
    MeteorMetric = env["meteor_metric"].MeteorMetric
    
    # Initialize the MeteorMetric
    meteor_scorer = MeteorMetric()
    
    scores = []
    
    for hyp, ref in zip(hyps, refs):
        # Evaluate each hypothesis against its reference
        score_dict = meteor_scorer.evaluate_example(hyp, [ref])  # Assuming evaluate_example expects a list of references
        scores.append(score_dict['meteor'])
    
    # Aggregate the scores into a single dictionary
    aggregated_scores = {
        "meteor_mean_score": sum(scores) / len(scores) if scores else 0
    }
    
    return aggregated_scores


@patch({
    "meteor":
    "data/packages/salaniz_pycocoevalcap/pycocoevalcap-1.2/meteor/meteor.py",
    "tokenizer":
        "data/packages/salaniz_pycocoevalcap/pycocoevalcap-1.2/tokenizer/ptbtokenizer.py"
})

def meteor_package_salaniz_pycocoevalcap(env, refs, hyps):
    meteor_scorer = env['meteor'].Meteor()
    tokenizer = env["tokenizer"].PTBTokenizer()
    
    # Ensure all captions are strings and prepare them for tokenization
    gts = {i: [{'caption': str(ref)}] for i, ref in enumerate(refs)}
    res = {i: [{'caption': str(hyp)}] for i, hyp in enumerate(hyps)}

    # Tokenize the ground truths and hypotheses
    gts_tokenized = tokenizer.tokenize(gts)
    res_tokenized = tokenizer.tokenize(res)

    # Compute the METEOR score for each tokenized pair of reference and hypothesis
    scores = []
    for i in range(len(refs)):
        score, _ = meteor_scorer.compute_score({i: gts_tokenized[i]}, {i: res_tokenized[i]})
        scores.append(score)

    # Aggregate the scores into a single dictionary
    aggregated_scores = {
        "meteor_mean_score": sum(scores) / len(scores) if scores else 0
    }

    return aggregated_scores

def meteor_package_nltk(refs, hyps):
    from nltk.translate.meteor_score import meteor_score
    scores = []

    # Ensure that both refs and hyps are lists of tokenized texts
    tokenized_refs = [[nltk.word_tokenize(ref) for ref in refs_i] for refs_i in refs]
    tokenized_hyps = [nltk.word_tokenize(hyp) for hyp in hyps]

    for ref, hyp in zip(tokenized_refs, tokenized_hyps):
        # Calculate METEOR score for each hypothesis against its references
        score = meteor_score(ref, hyp)
        scores.append({'meteor': score})
    
    # Aggregate scores if needed or return individual scores
    mean_score = sum(score['meteor'] for score in scores) / len(scores)
    return {'meteor_mean_score': mean_score}
    

@cache()

def run_configs_baseline() -> dict[str, float]:

    """

    Run baseline configuration.

    """

    return rouge_reference_individual(*meteor_data_dev(), args = "-n 2")

@cache()

def run_configs_apply_stemming() -> dict[str, float]:

    """

    Run baseline configuration plus stemming.

    """

    return rouge_reference_individual(*meteor_data_dev(), args = "-n 2 -m")

@cache()

def run_configs_remove_stopwords() -> dict[str, float]:

    """

    Run baseline configuration plus stopword removal.

    """

    return rouge_reference_individual(*meteor_data_dev(), args = "-n 2 -s")

@cache()

def run_configs_no_sentence_splits() -> list[dict[str, float]]:

    """

    Run baseline configuration, but remove all sentence splits.

    CNN / Daily Mail (and reference ROUGE) represents sentence splits using
    newlines. So, we simply remove all newlines. This causes ROUGE-L to treat
    huge multi-sentence model outputs as a single sentence.

    """

    refs, hyps = meteor_data_dev()

    refs = [" ".join(_.split()).strip() for _ in refs]
    hyps = [" ".join(_.split()).strip() for _ in hyps]

    return rouge_reference_individual(refs, hyps, args = "-n 2")

@cache()

def run_configs_period_sentence_splits() -> list[dict[str, float]]:

    """

    Run baseline configuration, but delete all sentence splits and resplit
    sentences using periods. This is similar to how some ROUGE packages operate,
    like pltrdy_files2rouge.

    """

    refs, hyps = meteor_data_dev()

    refs = [" ".join(_.split()).strip() for _ in refs]
    hyps = [" ".join(_.split()).strip() for _ in hyps]

    space = True

    if space:
        period = " ."
    else:
        period = "."

    refs = [_.replace(period, ".\n").replace("\n ", "\n").strip() for _ in refs]
    hyps = [_.replace(period, ".\n").replace("\n ", "\n").strip() for _ in hyps]

    return rouge_reference_individual(refs, hyps, args = "-n 2")

@cache()

def run_configs_nltk_sentence_splits() -> list[dict[str, float]]:

    """

    Run baseline configuration, but delete all sentence splits and resplit
    sentences using NLTK. This is similar to how some ROUGE packages operate,
    like bheinzerling_pyrouge (and several other pyrouges).

    """

    import nltk

    refs, hyps = meteor_data_dev()

    refs = [" ".join(_.split()).strip() for _ in refs]
    hyps = [" ".join(_.split()).strip() for _ in hyps]

    refs = ["\n".join(nltk.sent_tokenize(_)) for _ in refs]
    hyps = ["\n".join(nltk.sent_tokenize(_)) for _ in hyps]

    return rouge_reference_individual(refs, hyps, args = "-n 2")

@cache()

def run_configs_nltk_tokenize() -> list[dict[str, float]]:

    """

    Run baseline configuration, but pretokenize using NLTK.

    """

    import nltk

    refs, hyps = meteor_data_dev()

    refs = [
        "\n".join(
            " ".join(nltk.word_tokenize(line, preserve_line = True))
            for line in text.split("\n")
        )
        for text in refs
    ]

    hyps = [
        "\n".join(
            " ".join(nltk.word_tokenize(line, preserve_line = True))
            for line in text.split("\n")
        )
        for text in hyps
    ]

    return rouge_reference_individual(refs, hyps, args = "-n 2")

@cache()

def run_configs_truncate_75_bytes() -> list[dict[str, float]]:

    """

    Run baseline configuration with 75 byte truncation.

    """

    return rouge_reference_individual(*meteor_data_dev(), args = "-n 2 -b 75")

@cache()

def run_configs_truncate_100_words() -> list[dict[str, float]]:

    """

    Run baseline configuration with 100 word truncation.

    """

    return rouge_reference_individual(*meteor_data_dev(), args = "-n 2 -l 100")

@cache()

def run_configs_fscore_beta12() -> list[dict[str, float]]:

    """

    Run baseline configuration, but compute F_{1.2} scores instead of a more
    traditional balanced F_{1} score. This is inspired by tylin_cococaption,
    which has for many years accidentally computed incorrect F-scores.

    Note that Perl ROUGE actually represents F-scores using "alpha" rather than
    "beta." See paper for this discussion. Conversion equations included below.

    """

    beta = 1.2
    alpha = 1 / (1 + beta**2)

    return rouge_reference_individual(*meteor_data_dev(), args = f"-n 2 -p {alpha}")

@cache()

def run_configs_misreport_recall() -> list[dict[str, float]]:

    """

    Run baseline configuration. But, instead of reporting F-score, report
    full-length recall instead!

    """

    results = rouge_reference_individual(*meteor_data_dev(), args = "-n 2")

    for r in results:
        for k in ["1", "2", "l"]:

            (
                r[f"rouge-{k}-f"],
                r[f"rouge-{k}-r"],
            ) = (
                r[f"rouge-{k}-r"],
                r[f"rouge-{k}-f"],
            )

    return results

@cache()

def run_packages_yale_summeval()-> list[dict[str, float]]:
    refs, hyps = meteor_data_dev()  
    return meteor_package_yale_summeval(refs, hyps)

@cache()

def run_packages_nlg_eval()-> list[dict[str, float]]:
    refs, hyps = meteor_data_dev()  
    return meteor_package_nlg_eval(refs, hyps)

@cache()

def run_packages_salaniz_pycocoevalcap()-> list[dict[str, float]]:
    refs, hyps = meteor_data_dev()  
    return meteor_package_salaniz_pycocoevalcap(refs, hyps)

#@cache()
# FIXME
# def run_packages_nltk() -> dict[str, float]:
#     refs, hyps = meteor_data_dev()  # This function is reused; it provides suitable refs and hyps
#     return meteor_package_nltk(refs, hyps)

@cache()

def run_models_lead_3() -> dict[str, float]:

    """

    Evaluate Lead-3 using reference ROUGE.

    """

    return rouge_reference_overall(*meteor_data_test(), args = "-n 2 -m")

def generate_inline_data() -> None:

    """

    Generates non-tabular paper numbers.

    For example, "We find that 84% of people love dogs." contains an inline
    (non-tabular) number "84" which would be generated here.

    """

    import pandas as pd

    labels = load_dataset()
    rouge = labels.loc[labels["paper_rouge"]]
    acl_rouge = rouge.loc[rouge["paper_venue"] == "acl"]

    eval_packs = {
        "tylin_cococaption",

    }

    print("\n".join(_.strip() for _ in f"""

    ========================================
    Additional Data Variables
    ========================================

    Total number of papers with incorrect packages:

    {rouge["software_error"].sum()}

    Top ten packages by number of citations:

    {str(pd.Series([
        pack
        for packs in rouge.loc[rouge["software_error"], "packages"]
        for pack in packs
    ]).value_counts().iloc[:30]).replace("dtype: int64", "")}

    Average number of citations for each package evaluated

    {pd.Series([
        pack
        for packs in rouge.loc[rouge["software_error"], "packages"]
        for pack in packs
        if pack in eval_packs
    ]).value_counts().mean().round()}

    """.split("\n")))

def generate_overview_figure() -> None:

    """

    Generate Figure 1.

    """

    labels = load_dataset()
    meteor = labels.loc[labels["paper_meteor_prelim"] == True]

    print("\n".join(_.strip() for _ in f"""

    ===================
    (A) REPRODUCIBILITY
    ===================

    {len(meteor)} model evaluations using METEOR
    {meteor["reproducible"].mean() * 100:.0f}% reproducible

    (NOTE: see paper for details on comparison studies)
    
    =================
    (B) COMPARABILITY
    =================

    # Release code -- including incomplete and nonfunctional
    {(meteor["code_meteor_url"].apply(lambda x: bool(x))).mean() * 100:.0f}% papers
    
    Release code with METEOR evaluation
    {meteor["code_meteor"].mean() * 100:.0f}% papers
    
    List METEOR configuration parameters
    {(meteor["paper_meteor_params"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).mean() * 100):.0f}% papers
    
    Cite METEOR software package -- including unofficial
    {(meteor["paper_meteor_packages"].map(len) != 0).mean() * 100:.0f}% papers
    
    """.split("\n")))
    
    # ===============
    # (C) CORRECTNESS
    # ===============

    # Percentage of ROUGE software citations
    # that reference software with scoring errors
    # {rouge["software_error"].sum() / (rouge["packages"].map(len) != 0).sum() * 100:.0f}% papers

    

def generate_historical_plot():

    """

    Generate Figure 3.

    """

    import numpy as np
    import matplotlib.pyplot as plt

    data = load_dataset()

    data = data.loc[data["paper_meteor_prelim"]==True]

    data["uses_meteor"] = data["paper_meteor_prelim"]
    data["cites_package"] = data["paper_meteor_packages"].map(len) != 0
    data["has_error"] = data["software_error"]
    data["is_reproducible"] = data["reproducible"]

    result = (
        data
        .loc[data["paper_year"] >= 2004]
        .groupby("paper_year")
        [[
            "uses_meteor",
            "cites_package",
            "has_error",
            "is_reproducible"
        ]]
        .sum()
    )

    plt.rcParams["xtick.top"] = False
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["font.family"] = "sans-serif"

    plt.rcParams["figure.figsize"] = (8,5.5)
    years = list(result.index)
    plt.xticks(years, rotation=90)

    plt.bar(years, result["uses_meteor"], color = "lightgray", width=0.97)
    plt.bar(years, result["cites_package"], color = (0.60, 0, 0.05), width=0.97)
    plt.bar(years, result["cites_package"] - result["has_error"], color = (0.0, 0.6, 0.3), width=0.97)

    total_correct_package = (result["cites_package"] - result["has_error"]).sum()
    total_incorrect_package = (result["has_error"]) .sum()
    total_no_package = (result["uses_meteor"] - result["cites_package"]).sum()

    plt.text(2004, 610, f"⬤ No Package Citation (n = {total_no_package:,d})", size = 10, weight = "demibold", color = "gray")
    plt.text(2004, 580, f"⬤ Cites Incorrect Package (n = {total_incorrect_package:,d})", size = 10, weight = "demibold", color = (0.60, 0, 0.05))
    plt.text(2004, 550, f"⬤ Cites Correct Package (n = {total_correct_package:,d})", size = 10, weight = "demibold", color = (0.0, 0.6, 0.3))

    plt.xlim(2003.5, 2022.5)

    plt.grid(axis = "y", color = "white", linewidth = 1.2)
    plt.ylabel("Papers Performing METEOR Evaluation", weight = "demibold")

    plt.tight_layout()
    plt.savefig("correctness.pdf", bbox_inches = "tight", transparent=True)
    plt.close()

    plt.rcParams["xtick.top"] = True
    plt.rcParams["xtick.bottom"] = False
    plt.rcParams["font.family"] = "sans-serif"

    plt.rcParams["figure.figsize"] = (8,5.5)
    years = list(result.index)
    plt.xticks(years, rotation=90, color = "white")

    plt.bar(years, result["uses_meteor"], color = "lightgray", width=0.97)
    plt.bar(years, result["is_reproducible"], color = (0.05, 0.3, 0.5), width=0.97)

    total_reproducible = result["is_reproducible"].sum()
    total_not_reproducible = (result["uses_meteor"] - result["is_reproducible"]).sum()

    plt.text(2004, 585, f"⬤ Meets Basic Reproducibility Criteria (n = {total_reproducible:,d})", size = 10, weight = "demibold", color = (0.05, 0.3, 0.5))
    plt.text(2004, 615, f"⬤ Fails Basic Reproducibility Criteria (n = {total_not_reproducible:,d})", size = 10, weight = "demibold", color = "gray")

    plt.xlim(2003.5, 2022.5)

    plt.gca().invert_yaxis()

    plt.grid(axis = "y", color = "white", linewidth = 1.2)
    plt.ylabel("Papers Performing METEOR Evaluation", weight = "demibold")

    plt.tight_layout()
    plt.savefig("reproducibility.pdf", bbox_inches = "tight", transparent=True)
    plt.close()

def generate_process_figure() -> None:

    """

    Generate Figure 4.

    """

    papers = load_dataset()

    full = (
        papers
        .loc[
            (papers["paper_year"] >= 2002)
            &
            (~papers["error_download"])
            &
            (~papers["error_extract"])
        ]
    )

    prelim = (
        papers
        .loc[papers["paper_rouge_prelim"]]
    )

    rouge = (
        prelim
        .loc[prelim["paper_rouge"]]
    )

    code = (
        rouge
        .loc[rouge["code_url"] != ""]
    )

    print("\n".join(_.strip() for _ in f"""

    Overall Citations Collected
    ===========================

    Total Citations: {len(papers)}

    Download and Extract Text
    =========================

    Before 2002: {len(papers.loc[papers["paper_year"] < 2002])}
    Paper Inaccessible: {len(papers.loc[
        (papers["paper_year"] >= 2002)
        &
        papers["error_download"]
    ])}
    Extraction Errors: {len(papers.loc[
        (papers["paper_year"] >= 2002)
        &
        papers["error_extract"]
    ])}
    ----------
    Citations Excluded: {len(papers.loc[
        (papers["paper_year"] < 2002)
        |
        papers["error_download"]
        |
        papers["error_extract"]
    ])}

    Full-Text Machine Learning Papers
    =================================

    {len(full)}

    Screen Papers for ROUGE
    =======================

    Automated Rules: {len(full) - len(prelim)}
    ----------
    Papers Excluded: {len(full) - len(rouge)}

    ROUGE Papers Included in Review
    ===============================

    {len(rouge)}

    Screen Code for ROUGE
    =====================

    Code Unavailable: {len(rouge.loc[
        (rouge["code_url"] == "")
        &
        (rouge["paper_venue"] == "acl")
    ])}
    Linking Errors: {len(rouge.loc[
        (rouge["code_url"] == "")
        &
        (rouge["paper_venue"] != "acl")
    ])}
    ----------
    Codebases Excluded: {len(rouge) - len(code)}

    ROUGE Codebases Included in Review
    ==================================

    {len(code)}

    """.split("\n")))

def generate_configs_table() -> None:

    """

    Generate Table 1.

    """

    import concurrent.futures as cf
    from tqdm import tqdm

    experiments = collect("^run_configs")

    ex = cf.ProcessPoolExecutor()

    results = {
        exp.__name__.replace("run_configs_", ""):
        ex.submit(exp) for exp in experiments
    }

    list(tqdm(
        cf.as_completed(results.values()),
        total = len(results),
        desc = "Protocol Experiments"
    ))

    df = pd.DataFrame([
        pd.DataFrame(v.result()).mean().rename(k) * 100
        for k, v in results.items()
    ])

    delta = df - df.loc["baseline"]

    recall_experiments = df.index.str.startswith("truncate_")
    fscore_experiments = ~recall_experiments & (df.index != "baseline")

    print("\n" + "=" * 80)
    print("Rogue Scores Table 1:\nComparability Experiments\n")

    print(pd.concat([

        delta
        .loc[
            fscore_experiments,
            ["rouge-1-f", "rouge-2-f", "rouge-l-f"],
        ]
        .rename(axis = 1, mapper = lambda _: _.replace("-f", "").upper())
        .round(2),

        delta
        .loc[
            recall_experiments,
            ["rouge-1-r", "rouge-2-r", "rouge-l-r"],
        ]
        .rename(axis = 1, mapper = lambda _: _.replace("-r", "").upper())
        .round(2)

    ]))

    print("=" * 80 + "\n")

def generate_packages_table() -> None:

    """

    Generate Table 2.

    """

    import concurrent.futures as cf
    from tqdm import tqdm

    meteor_install_packages()

    experiments = collect("^run_packages")

    results = {
        exp.__name__.replace("run_packages_", ""):
        exp() for exp in tqdm(
            experiments,
            total = len(experiments),
            desc = "Correctness Experiments"
        )
    }

    data_for_df = []

    for package, scores in results.items():
        if isinstance(scores, dict):
            scores = [scores]
        
        for score_dict in scores:
            score_dict['Package'] = package
            data_for_df.append(score_dict)
            
        df = pd.DataFrame(data_for_df)
        
        cols = ['Package'] + [col for col in df if col != 'Package']
        df = df[cols]

    
    print("\n" + "=" * 80)
    print("METEOR Scores Table 2:\nCorrectness Experiments\n")

    print(df)

    print("=" * 80)

def generate_models_table() -> None:

    """

    Generate Table 3.

    """

    import concurrent.futures as cf
    from tqdm import tqdm

    experiments = collect("^run_models")

    ex = cf.ProcessPoolExecutor()

    futures = {
        exp.__name__.replace("run_models_", ""):
        ex.submit(exp) for exp in experiments
    }

    list(tqdm(
        cf.as_completed(futures.values()),
        total = len(futures),
        desc = "Case Study Experiments"
    ))

    df = pd.DataFrame({k: v.result() for k, v in futures.items()}).T * 100

    print("\n" + "=" * 80)
    print("Rogue Scores Table 3:\nRogue-3 Case Study\n")

    print(
        df[["rouge-1-f", "rouge-2-f", "rouge-l-f"]]
        .rename(axis = 1, mapper = lambda _: _.replace("-f", "").upper())
        .round(2)
    )

    print("=" * 80 + "\n")

def load_dataset() -> pd.DataFrame:

    """

    Load the dataset release.

    """

    try: return pd.read_json(
        "data/meteor_papers.jsonl.gz",
        orient = "records", lines = True)
    except: pass

    print("Could not load dataset.")

