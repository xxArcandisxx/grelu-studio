# Inputs derived from the UCSC hg38 sample

Reference DNA is real hg38 sequence from test_10.fasta. Alternate alleles and
variant CSV entries are constructed substitutions for software testing, not
observed GM12878 variants. Exact motif patterns are extracted from a reference
sequence and are not annotated transcription-factor motifs.

Training/fine-tuning labels are synthetic GC fractions, not measured DNase
activity. Training windows are center-cropped to 2,000 bases (57-base offset);
fine-tuning retains all 2,114 bases. These check optimizer/checkpoint execution
only and do not establish biological model accuracy.

PWM testing uses the existing explicitly synthetic JASPAR-format fixture.
