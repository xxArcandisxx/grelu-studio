"""Compare user-supplied, aligned regulatory sequence windows."""
MODES = {
    "allele-specific": (
        "Allele-Specific Regulation",
        "Provide exactly two aligned FASTA records from the same locus. The first is the baseline haplotype.",
    ),
    "prioritize": (
        "Variant Prioritization",
        "Provide a reference as the first FASTA record, followed by up to 11 supplied single-substitution variants at that locus.",
    ),
    "multi-allelic": (
        "Multi-Allelic & Combinatorial",
        "Provide a reference as the first FASTA record, followed by up to 11 supplied alternate sequences for the same locus.",
    ),
}


def validate_pairs(mode, records):
    if len(records) < 2 or (mode == "allele-specific" and len(records) != 2):
        raise ValueError("Provide exactly two FASTA records." if mode == "allele-specific"
                         else "Provide a reference and at least one alternate FASTA record.")
    reference = records[0]["sequence"]
    counts = {}
    for index, row in enumerate(records[1:], 2):
        differences = 0
        for ref, alt in zip(reference, row["sequence"], strict=True):
            if ref != alt:
                if "N" in (ref, alt):
                    raise ValueError(f"Record {index} contains a difference involving an unknown base (N).")
                differences += 1
        if not differences:
            raise ValueError(f"Record {index} is identical to the first sequence.")
        if mode == "prioritize" and differences != 1:
            raise ValueError(f"Record {index} must differ from the reference at exactly one position.")
        counts[row["id"]] = differences
    return counts


def comparisons(mode, predictions, counts):
    baseline = predictions[0]
    output = []
    for track, reference_score in baseline["scores"].items():
        entries = []
        for alternate in predictions[1:]:
            score = alternate["scores"][track]
            reverse_delta = (alternate["reverse_complement_scores"][track]
                             - baseline["reverse_complement_scores"][track])
            entries.append({
                "track": track, "reference": baseline["id"], "alternate": alternate["id"],
                "differences": counts[alternate["id"]], "reference_score": reference_score,
                "alternate_score": score, "delta": score - reference_score,
                "reverse_complement_delta": reverse_delta,
            })
        if mode == "prioritize":
            entries.sort(key=lambda entry: -abs(entry["delta"]))
            for i, entry in enumerate(entries, 1):
                entry["rank_within_track"] = i
        output.extend(entries)
    return output
