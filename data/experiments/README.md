# High-dimensional attribution experiment datasets

These datasets let you measure attribution cost as the number of measured inputs
grows well beyond the number of independent observations. The largest prepared
Cox dataset has **511 patients and 396,065 methylation features**. All downloads
are retained, and the prepared arrays keep outcomes separate from predictors.

## Available data

Counts below are measured from the prepared files, after the exclusions described
below. OS means overall survival; an OS event is a death. A censored observation
has follow-up time but no observed death at that time.

| Loader name | Independent observations / provided split | Predictor dimensions | Endpoint |
| --- | ---: | ---: | --- |
| `tcga_lgg_methylation` | 511 patients; 125 deaths | **396,065** | OS in days; DSS, DFI, PFI also retained |
| `tcga_lusc` | 418 patients; 132 deaths | 100,892 | OS in days |
| `tcga_hnsc` | 443 patients; 152 deaths | 97,536 | OS in days |
| `tcga_lgg_multiomics` | 419 patients; 77 deaths | 90,151 | OS in days |
| `tcga_laml` | 35 patients; 14 deaths | 90,159 | OS in days |
| `gse24080` | 553 patients; 339 training / 214 validation; 168 deaths | 54,675 | OS in months; EFS also retained |
| `motorimagery` | 278 training / 100 test trials, **one subject** | 64 channels × 3,000 samples = 192,000 | Finger/tongue classification |
| `xjtu_sy` | **15 bearings**, 9,216 repeated recordings | 32,768 samples × 2 channels = 65,536 | Time to final recording, in minutes |

HNSC is head and neck squamous cell carcinoma; LUSC is lung squamous cell
carcinoma; LGG is lower-grade glioma; LAML is acute myeloid leukemia. The two LGG
datasets overlap in patients and should not be presented as independent cohorts.
The small LAML cohort is useful for computational stress tests, but has only 14
observed deaths for fitting and evaluating a survival model.

## Fit with the paper's Cox setting

To demonstrate the quadrature savings, retain many active model inputs and explain
the **relative hazard**, the multiplicative risk score in the paper. With fitted
coefficients `beta` and a processed observation `x`, this is `exp(x @ beta)`.
The log score `x @ beta` is additive, while survival probability at a fixed time
has a different functional form. Results for the relative hazard should be
labeled accordingly.

Use regularization to fit a Cox model when predictors outnumber patients. A ridge
penalty, which shrinks coefficients without deliberately setting them to zero,
can preserve many active dimensions. An L1 penalty can set most coefficients to
zero: report the number of nonzero coefficients as well as the input dimension,
because zero coefficients do not contribute to this product game. Constant
features removed using the training set also reduce the dimension.

For a model with `d_active` contributing features, exact Gauss–Legendre integration
needs at most `ceil(d_active / 2)` nodes in the paper's `p = 1` Cox construction.
Smaller node counts can therefore be tested on these real measurements. Use
background rows from the training set; replace missing coalition features using
each whole background row to match the paper's empirical-background game.

MotorImagery supplies high-dimensional classification inputs, not survival times.
XJTU-SY supplies observed run-to-failure histories. Its windows share a bearing
and must not be split randomly across training and evaluation. A helper below
constructs one observation per bearing, with the final recording as an explicit
operational failure endpoint; no censoring is invented.

## Load and preprocess

Run these examples from the repository root, with the project's virtual
environment. Arrays are NumPy `float32`; loaders use memory mapping so opening
an array does not first copy the whole file into RAM. Row selection and
preprocessing can allocate substantial additional memory.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from benchmarks.experiment_data import load_survival, TrainingPreprocessor

dataset = load_survival("tcga_lgg_methylation")
train, test = train_test_split(
    np.arange(len(dataset.y)), test_size=0.25, random_state=42,
    stratify=dataset.y["event"],
)
preprocessor = TrainingPreprocessor.fit(dataset.X[train])
X_train = preprocessor.transform(dataset.X[train])
X_test = preprocessor.transform(dataset.X[test])
y_train, y_test = dataset.y[train], dataset.y[test]
feature_names = dataset.feature_names[preprocessor.keep]
```

`y` has the structured fields `event` (Boolean, `True` means an event) and `time`
(positive duration), compatible with scikit-survival. A model library is not
installed by the dataset requirements, and no Cox model has been fitted here.
The helper learns medians, means, and standard deviations from training rows
only, removes features that are entirely missing or constant in training, and
reuses those statistics on evaluation rows. Fit it separately inside each
cross-validation fold.

For the provided myeloma split:

```python
myeloma = load_survival("gse24080", endpoint="OS")  # or endpoint="EFS"
train = myeloma.samples["split"].eq("training").to_numpy()
validation = myeloma.samples["split"].eq("validation").to_numpy()
```

For the time series:

```python
from benchmarks.experiment_data import (
    load_motorimagery, load_bearing, load_bearing_landmark,
)

X_train, labels_train, trials_train = load_motorimagery("train")  # (278, 192000)
X_test, labels_test, trials_test = load_motorimagery("test")      # (100, 192000)

signals, remaining_minutes, windows = load_bearing("Bearing1_1", flatten=False)
# signals.shape == (123, 32768, 2); remaining_minutes ends in zero.

X, y, bearings = load_bearing_landmark(window_index=1)
# X.shape == (15, 65536); y has positive follow-up and 15 observed events.
# Each row uses only the first recording of one bearing. Split by bearing.
```

MotorImagery flattens in channel-then-time order: feature `channel * 3000 + time`.
XJTU-SY flattens in time-then-channel order: feature `time * 2 + channel`, with
horizontal then vertical vibration. Keeping `flatten=False` preserves the axes.
All 15 bearings must have positive remaining time for `load_bearing_landmark`;
later recording indices that violate that requirement are rejected.

## Sources and preparation decisions

### TCGA multi-omics cohorts

Source: [Herrmann et al.'s benchmark repository](https://github.com/HerrMo/multi-omics_benchmark_study)
and [published benchmark study](https://doi.org/10.1093/bib/bbaa167).
The original ARFF files are downloaded from OpenML:

| Cohort | Part 1 | Part 2 |
| --- | --- | --- |
| HNSC | [42285](https://www.openml.org/d/42285) | [42286](https://www.openml.org/d/42286) |
| LUSC | [42299](https://www.openml.org/d/42299) | [42300](https://www.openml.org/d/42300) |
| LGG | [42293](https://www.openml.org/d/42293) | [42294](https://www.openml.org/d/42294) |
| LAML | [42291](https://www.openml.org/d/42291) | [42292](https://www.openml.org/d/42292) |

Each cohort combines RNA, microRNA, copy-number, mutation, and encoded clinical
predictors. The published table's total columns include patient barcode, time,
and status, so the predictor counts above are **three smaller**. All three
non-predictor columns are excluded from `X`.

Part 2 does not contain patient IDs. Preparation preserves the provider's
matching row order and checks row counts; it cannot independently verify the
join using IDs absent from that part. No rows are reordered. The provider's
preprocessing and clinical encoding are retained, including any filtering or
imputation applied upstream; these files do not establish that upstream
preprocessing was fitted within your eventual training split. OpenML metadata
records license `GPL-2` and is retained alongside each download.

### LGG methylation with clinical follow-up

Measurements: [UCSC Xena TCGA LGG HumanMethylation450](https://xenabrowser.net/datapages/?dataset=TCGA.LGG.sampleMap%2FHumanMethylation450&host=https%3A%2F%2Ftcga.xenahubs.net).
Outcomes: the TCGA Clinical Data Resource (CDR), Supplemental Table S1, from
[GDC's Pan-Cancer Atlas publication files](https://gdc.cancer.gov/about-data/publications/pancanatlas).
Its [clinical endpoint study](https://doi.org/10.1016/j.cell.2018.02.052) describes
the endpoint definitions and cohort-specific follow-up limitations.

The source has 530 profiles and 485,577 sites. Preparation matches the first 12
characters of each TCGA sample barcode to the CDR patient barcode, keeps primary
tumors (sample type `01`), and requires an LGG match and valid positive OS time.
The resulting 511 patients have unique IDs. The 19 excluded profiles and their
reasons are saved in `excluded_samples.tsv`.

Of the original sites, 89,512 are missing for every eligible patient and are
listed in `excluded_features.tsv`. Removing them leaves **396,065 sites** and
165,596 missing measurements. These remaining missing values are retained for
training-only imputation. Values are methylation proportions in `[0, 1]` (often
called methylation beta values; they are not Cox regression coefficients).

Disease-specific survival (DSS), disease-free interval (DFI), and progression-free
interval (PFI) are retained alongside OS. Selecting an alternate endpoint removes
rows with missing or nonpositive duration or invalid event coding. These are
subsets of the OS-eligible cohort; they are not separate cohorts optimized for
each endpoint. Baseline clinical fields are in `clinical_covariates.tsv`, separate
from the methylation matrix.

### GSE24080 multiple myeloma

Source: [NCBI GEO GSE24080](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24080),
using the processed series matrix and the supplied June 2008 clinical workbook.
The source contains 559 expression profiles and 565 clinical rows. Expression
sample titles are matched to clinical CEL filenames, with a one-to-one join.
Five provider-flagged `MAQC_Q` outliers and one zero-OS-duration patient are
excluded, leaving 553 patients and all 54,675 probe sets. Probe sets are not
aggregated into genes. Deposited expression processing is preserved.

Continuous June 2008 OS and event-free survival (EFS) durations are in **months**;
the event columns explicitly use `1=death` and `1=event`. The 24-month binary
classification labels are not used as survival targets. EFS has 245 events.
The provider's training/validation split and treatment protocol are retained.
Baseline clinical measurements are stored separately from `X`.

### MotorImagery

Source: the [UEA time-series archive's MotorImagery dataset](https://www.timeseriesclassification.com/description.php?Dataset=MotorImagery),
derived from [BCI Competition III, dataset I](https://www.bbci.de/competition/iii/desc_I.html).
These are electrocorticography recordings: electrical activity measured with an
electrode grid on the cortical surface. One subject imagines finger or tongue
movements. Each trial has 64 channels sampled at 1,000 Hz for three seconds.
The official 278/100 split uses sessions recorded about a week apart. Preserve
that split; the 378 trials are not 378 independent subjects.

### XJTU-SY bearings

Source description: [dataset creators' page](https://biaowang.tech/xjtu-sy-bearing-datasets/)
and [Wang et al.'s publication](https://doi.org/10.1109/TR.2018.2882682).
The archive was downloaded from the
[community mirror used by the rul-datasets reader](https://github.com/tilman151/rul-datasets/blob/master/rul_datasets/reader/xjtu_sy.py),
at `https://kr0k0tsch.de/rul-datasets/XJTU-SY.zip`. This is a mirror copy, not a
download from the creators' original file host. Its locally computed checksum
records the retrieved bytes; it is not an original-source authenticity signature.

All 15 bearing runs are retained. Conditions 1, 2, and 3 correspond respectively
to 35 Hz/12 kN, 37.5 Hz/11 kN, and 40 Hz/10 kN. Each recording contains 32,768
samples from two vibration channels at 25,600 Hz, and recordings occur at
one-minute intervals. Every CSV is parsed and checked for shape and finite values;
consecutive indices and expected run lengths are checked for all bearings.

The supplied target is **minutes to the final recorded window**, computed as
`number_of_recordings - one_based_window_index`. It is an operational proxy for
remaining life, with zero at the final recording, not an independently annotated
exact failure timestamp. The landmark loader selects one observed window per
bearing and requires positive remaining time for Cox experiments. There is no
provided censoring or official train/test split.

## Reproduce and inspect

```sh
uv pip install --python .venv/bin/python -r benchmarks/requirements-experiment-data.txt
.venv/bin/python benchmarks/fetch_experiment_data.py
.venv/bin/python benchmarks/prepare_experiment_data.py
.venv/bin/python benchmarks/validate_experiment_data.py
.venv/bin/python -m pytest tests/test_experiment_data.py -q
```

The first command assumes the repository's `.venv` already exists. The data
scripts need NumPy, pandas, openpyxl, xlrd, and the `curl` executable; the example
random split additionally uses scikit-learn. Download or prepare a subset with,
for example, `--datasets tcga_lgg_methylation motorimagery`. Downloads resume from
`.part` files and verify cached SHA-256 receipts. Preparation keeps completed
outputs unless `--force` is supplied. Raw files are never overwritten by
preparation.

- `raw/`: original archives/tables plus source URLs, sizes, and SHA-256 receipts.
- `processed/<dataset>/`: arrays, identifiers, labels, and metadata.
- `manifest.json`: source receipts and prepared dataset counts.
- `validation.json`: verification results and hashes of prepared files.

Large downloads and generated matrices are ignored by Git. Scripts, this guide,
the manifest, and validation report can be versioned. Checksums detect changes
relative to this download; they do not establish independent upstream authenticity.
Use the original dataset publications and provider terms when citing these data.

The complete local bundle occupies approximately **10.60 GiB**: 6.72 GiB of raw
files and 3.88 GiB of prepared data. The initial validation verified all 23 source
files, checked all eight prepared datasets, and recorded hashes for 52 prepared
files. The 10 loader/preprocessing unit tests passed.
