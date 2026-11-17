"""Small-data training and sequence motif analysis used by the lab pages."""
import csv
import io
import math
import random
from collections import Counter

import numpy as np


def training_example(length):
    rng = random.Random(23)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['sequence', 'label'])
    for _ in range(40):
        gc = rng.uniform(.15, .85)
        sequence = ''.join(rng.choices('ACGT', weights=[(1-gc)/2,gc/2,gc/2,(1-gc)/2], k=length))
        writer.writerow([sequence, (sequence.count('G') + sequence.count('C')) / length])
    return output.getvalue()


def training_rows(text, length):
    from studio_web import parse_sequence
    reader = csv.DictReader(io.StringIO(text), strict=True)
    if reader.fieldnames != ['sequence', 'label']:
        raise ValueError('Training CSV needs exactly sequence,label columns.')
    rows, seen = [], set()
    for index, row in enumerate(reader, 2):
        if len(rows) == 256:
            raise ValueError('Use at most 256 records per local training run.')
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'CSV row {index} has missing or extra fields.')
        sequence = parse_sequence(row['sequence'], f'row {index}')
        if len(sequence) != length:
            raise ValueError(f'Row {index} must contain {length:,} bases.')
        if sequence in seen:
            raise ValueError('Use distinct sequences so duplicates do not cross the held-out split.')
        label = float(row['label'])
        if not math.isfinite(label) or abs(label) > 1e6:
            raise ValueError('Labels must be finite numbers with magnitude at most 1,000,000.')
        rows.append((sequence, label))
        seen.add(sequence)
    if len(rows) < 8:
        raise ValueError('Provide at least eight distinct labeled sequences.')
    return rows


def train(job_id, rows, spec, epochs, learning_rate, target, data_source="uploaded numeric labels"):
    import torch
    import hashlib
    from core.models import load_model
    from core.sequences import one_hot
    from lab_jobs import job_path, update_job
    from model_registry import registrations, save_registration
    from studio_web import inference_lock
    from flask import current_app

    with inference_lock:
        torch.manual_seed(0)
        batch_size = int(current_app.config.get('TRAIN_BATCH_SIZE',8))
        if not 1 <= batch_size <= 32:
            raise ValueError('Training batch size must be between 1 and 32.')
        adapter, _ = load_model(spec, 'cpu', batch_size=batch_size, backend='builtin')
        module = adapter.module
        if sum(parameter.numel() for parameter in module.parameters()) > 25_000_000:
            raise ValueError('This local CPU trainer supports models up to 25 million parameters.')
        if not 0 <= target < len(adapter.task_names):
            raise ValueError('Choose a valid output track index.')
        permutation = np.random.default_rng(0).permutation(len(rows))
        count = max(2, round(len(rows) * .2))
        validation_indices, train_indices = permutation[:count], permutation[count:]
        x = torch.from_numpy(one_hot([sequence for sequence, _ in rows], adapter.seq_len))
        y = torch.tensor([label for _, label in rows], dtype=torch.float32)
        def prediction(batch):
            output = module(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output[:, target].reshape(len(batch), -1).mean(dim=1)
        def validation():
            module.eval()
            total_error = 0.0
            with torch.no_grad():
                for offset in range(0,len(validation_indices),batch_size):
                    indices = validation_indices[offset:offset+batch_size]
                    total_error += float(torch.nn.functional.mse_loss(prediction(x[indices]),y[indices],reduction='sum'))
            return total_error/len(validation_indices)
        initial = validation()
        best, best_state = initial, {key:value.detach().clone() for key,value in module.state_dict().items()}
        optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
        history = []
        optimizer_steps, best_epoch = 0, 0
        for epoch in range(1, epochs + 1):
            module.train()
            losses = []
            for offset in range(0, len(train_indices), batch_size):
                indices = train_indices[offset:offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(prediction(x[indices]), y[indices])
                if not torch.isfinite(loss):
                    raise ValueError('Training became non-finite. Reduce the learning rate or rescale labels.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
                losses.append((float(loss.detach()), len(indices)))
            val = validation()
            if not math.isfinite(val):
                raise ValueError('Validation became non-finite. Reduce the learning rate.')
            if val < best:
                best = val
                best_epoch = epoch
                best_state = {key:value.detach().clone() for key,value in module.state_dict().items()}
            history.append({'epoch': epoch, 'training_loss': sum(loss*n for loss,n in losses)/sum(n for _,n in losses), 'validation_loss': val})
            update_job(job_id, message=f'Epoch {epoch} of {epochs}', history=history)
        module.load_state_dict(best_state)
        path = job_path(job_id) / 'model.pt'
        torch.save({'state_dict': best_state, 'base_model': spec, 'task_names': adapter.task_names}, path)
        key = 'local-' + job_id[:12]
        save_registration(key, {'source':'studio_trained','path':str(path),'seq_len':adapter.seq_len},
                          f'Local trained model · {job_id[:8]}', 'training')
        source_digest = None
        if spec.get('path'):
            with open(spec['path'],'rb') as source_file:
                source_digest = hashlib.file_digest(source_file,'sha256').hexdigest()
        baseline = float(((y[validation_indices] - y[train_indices].mean()) ** 2).mean())
        return {'model_key':key, 'initial_validation_mse':initial, 'best_validation_mse':best,
                'mean_label_baseline_mse':baseline, 'training_records':len(train_indices), 'validation_records':count,
                'target_track':adapter.task_names[target], 'epochs':epochs, 'learning_rate':learning_rate,
                'optimizer_steps':optimizer_steps,'best_epoch':best_epoch,'batch_size':batch_size,
                'model_provenance':registrations()[key]['provenance'],
                'base_model_config':{name:value for name,value in spec.items() if name!='path'},
                'base_checkpoint_sha256':source_digest,
                'split_seed':0, 'data_source':data_source, 'checkpoint':'model.pt', 'history':history}


def motif_analysis(records, patterns, k):
    from core.sequences import revcomp
    hits = []
    for record in records:
        sequence = record['sequence']
        for name, motif in patterns:
            orientations = [('+', motif)]
            if revcomp(motif) != motif:
                orientations.append(('-', revcomp(motif)))
            for strand, query in orientations:
                for start in range(len(sequence) - len(query) + 1):
                    if sequence[start:start+len(query)] == query:
                        hits.append({'record':record['id'],'motif':name,'sequence':query,'start':start+1,'end':start+len(query),'strand':strand})
                        if len(hits) > 10000:
                            raise ValueError('More than 10,000 hits. Use fewer sequences or longer motifs.')
    observed = Counter()
    background = Counter()
    rng = random.Random(0)
    from core.sequences import dinuc_shuffle
    for record in records:
        sequence = record['sequence']
        shuffled = dinuc_shuffle(sequence, rng)
        for index in range(len(sequence)-k+1):
            word = sequence[index:index+k]
            if 'N' not in word:
                observed[min(word, revcomp(word))] += 1
            word = shuffled[index:index+k]
            if 'N' not in word:
                background[min(word, revcomp(word))] += 1
    discovery = [{'kmer':word,'count':count,'background_count':background[word],
                  'enrichment':(count+1)/(background[word]+1)} for word,count in observed.items() if count >= 2]
    discovery.sort(key=lambda row:(-row['enrichment'],-row['count'],row['kmer']))
    return {'hits':hits,'discovery':discovery[:30],'records':len(records),'k':k,'patterns':patterns,
            'method':'Exact sequence motifs on both strands; exploratory k-mer enrichment against one dinucleotide-shuffled background per sequence.'}
