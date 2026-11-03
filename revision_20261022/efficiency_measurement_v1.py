"""Separate runtime measurements, never accuracy/model-selection experiments."""

import gc
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace

TASKS = ('splice_sites_all', 'splice_sites_acceptor', 'splice_sites_donor', 'H3K27me3')
SEEDS = (42, 2024, 3407)
CONDITIONS = (('local_only', False), ('local_comp', False), ('full', False),
              ('local_comp', True), ('full', True))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol():
    return dict(kind='actual_efficiency_measurements_v1', tasks=list(TASKS), seeds=list(SEEDS),
        conditions=[dict(variant=v, teacher=t) for v, t in CONDITIONS],
        train_rows=27000, inference_rows=3000, training_epochs_measured=1,
        warmup_optimizer_updates=3, inference_warmup_batches=5, inference_passes=3,
        latency_rows=64, batch_size=16, channels_per_kernel=128, kernels=[3, 5, 7, 11, 15],
        lr=.0001, weight_decay=.01, amp=True, teacher_alpha=.5, teacher_temperature=2.,
        teacher_epoch_seed=42, teacher_microbatch=4, teacher_accumulation=4,
        teacher_epoch_lr=.00001, feature_batch_size=16, feature_precision='original_FP32',
        test_rows_consumed=0, accuracy_results=0, automatic_retry=False,
        training_scope='One full steady-state epoch after three warmup updates;not a full15epoch fit.',
        inference_scope='Cached inputs,H2D,forward,float CPU copy;no scoring or metric computation.',
        cost_scope='Fresh train/dev context extraction,warm cache reads,teacher epoch and logit extraction separately.',
        original_teacher_cost_scope='Observed full-run wall costs,including sharing/possible pauses;not a controlled speed comparison.')


def case_order(seed):
    index = SEEDS.index(seed)
    # Cyclic order reduces systematic early/late bias without score-based order.
    return list(CONDITIONS[index:] + CONDITIONS[:index])


def duration_summary(seconds, rows):
    require(len(seconds) > 0 and all(math.isfinite(v) and v > 0 for v in seconds), 'Invalid timing')
    return dict(seconds=list(seconds), mean_seconds=statistics.mean(seconds),
        sample_SD_seconds=statistics.stdev(seconds) if len(seconds) > 1 else None,
        rows_per_pass=rows, sequences_per_second=[rows / v for v in seconds])


def validate_task_result(result, task):
    require(task in TASKS and result['task'] == task and result['protocol'] == protocol(), 'Timing protocol differs')
    require(result['accuracy_results'] == result['test_results'] == 0, 'Timing must not contain accuracy evidence')
    rows = result['cases']
    wanted = {(s, v, t) for s in SEEDS for v, t in CONDITIONS}
    require(len(rows) == 15 and {(r['seed'], r['variant'], r['teacher']) for r in rows} == wanted,
            'Missing or duplicate timing case')
    for row in rows:
        require(row['training']['rows'] == 27000 and row['training']['optimizer_updates'] == math.ceil(27000/16),
                'Not a complete training epoch')
        require(math.isfinite(row['training']['seconds']) and row['training']['seconds'] > 0, 'Invalid epoch time')
        require(row['inference']['rows_per_pass'] == 3000 and len(row['inference']['seconds']) == 3,
                'Incomplete inference measurement')
        expected = duration_summary(row['inference']['seconds'], 3000)
        require(row['inference'] == expected, 'Inference summary arithmetic differs')
        require(len(row['latency_seconds']) == 64 and all(math.isfinite(x) and x > 0 for x in row['latency_seconds']),
                'Invalid latency samples')
        for key in ('training_memory', 'inference_memory'):
            require(row[key]['peak_allocated_bytes'] > 0 and row[key]['peak_reserved_bytes'] >= row[key]['peak_allocated_bytes'],
                    'Invalid CUDA allocator peaks')
    require(set(result['features']) == {'train', 'dev'} and all(not v['cold']['reused'] for v in result['features'].values()),
            'Cold feature extraction was not measured')
    require(all(v['warm']['reused'] and v['cold']['array_sha256'] == v['warm']['array_sha256']
                for v in result['features'].values()), 'Warm cache integrity differs')
    teacher = result['teacher_epoch']
    require(teacher['rows'] == 27000 and teacher['optimizer_updates'] == math.ceil(27000/16)
            and math.isfinite(teacher['seconds']) and teacher['seconds'] > 0, 'Incomplete teacher epoch')
    require(result['teacher_logits']['rows'] == 27000 and result['teacher_logits']['freshly_extracted'] is True,
            'Teacher extraction was not measured')
    require(result['monitor_samples'] and not result['monitor_errors'], 'Resource monitoring incomplete')
    return True


def measure_task(plan, task, directory, dependencies, gpu_module):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = Path(plan['root'])
    here = root / 'revision_20261022'
    sys.path.insert(0, str(here))
    import run_controlled_experiments as control
    import run_fair_baselines as fair
    import run_teacher_ablation as teacher
    require(task in TASKS and plan['protocol'] == protocol(), 'Unknown measurement task')
    device = torch.device('cuda:0')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one bound CUDA device required')
    torch.set_num_threads(4)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    options = SimpleNamespace(smoke=False, split_dir=root / 'data/splits/nt_tasks',
        model_dir=root / 'models/dnabert2', context_cache=directory / 'fresh_context', device=device, amp=True)
    case = fair.TaskCase(options, task)
    require({s: len(r) for s, r in case.rows.items()} == {'train': 27000, 'dev': 3000}, 'Dataset row counts differ')
    require(case.hashes == plan['splits'][task], 'Dataset hashes changed')
    samples, monitor_errors = [], []
    stopped = threading.Event()

    def monitor():
        while not stopped.is_set():
            try:
                samples.append(gpu_module.observe())
            except BaseException as error:
                monitor_errors.append(repr(error))
            stopped.wait(5.)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    def sync():
        torch.cuda.synchronize(device)

    def memory():
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device))

    def clear():
        gc.collect()
        torch.cuda.empty_cache()

    def measure_epoch(model, loader, teacher_mode=False, micro=16):
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
            lr=.00001 if micro == 4 else .0001, weight_decay=.01)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        weights = case.weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='sum' if micro == 4 else 'mean')

        def update(group):
            optimizer.zero_grad(set_to_none=True)
            denominator = sum(float(case.weights[b[1]].sum()) for b in group) if micro == 4 else 1.
            total = 0
            for batch in group:
                with control.amp_context(device, True):
                    if micro == 4:
                        inputs, labels, _ = batch
                        logits = fair.forward(model, 'dnabert_finetune', fair.to_device(inputs, device))
                        loss = criterion(logits, labels.to(device)) / denominator
                    else:
                        bases, comp, context, labels = batch[:4]
                        logits = model({}, bases.to(device), comp.to(device), bert_pooled=context.to(device))
                        loss = criterion(logits, labels.to(device))
                        if teacher_mode:
                            loss = loss + .5 * teacher.distillation_loss(logits, batch[4].to(device), temperature=2.)
                    require(bool(torch.isfinite(loss)), 'Nonfinite benchmark loss;retain failed attempt')
                scaler.scale(loss).backward()
                total += len(labels)
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            return total

        accumulation = 16 // micro
        warm = iter(loader)
        for _ in range(3):
            update(list(itertools.islice(warm, accumulation)))
        sync()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        iterator, count, updates = iter(loader), 0, 0
        while True:
            group = list(itertools.islice(iterator, accumulation))
            if not group:
                break
            count += update(group)
            updates += 1
        sync()
        result = dict(rows=count, optimizer_updates=updates, seconds=time.perf_counter() - start,
                      warmup_optimizer_updates=3)
        peak = memory()
        del optimizer, scaler, warm
        return result, peak

    def infer(model, dataset, batch_size, repetitions):
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        def forward(batch):
            bases, comp, context, _ = batch[:4]
            with control.amp_context(device, True):
                values = model({}, bases.to(device), comp.to(device), bert_pooled=context.to(device))
            return values.float().cpu()

        with torch.inference_mode():
            for batch in itertools.islice(iter(loader), 5):
                forward(batch)
            sync()
            torch.cuda.reset_peak_memory_stats(device)
            times, checksums = [], []
            for _ in range(repetitions):
                start, outputs = time.perf_counter(), []
                for batch in loader:
                    outputs.append(forward(batch))
                sync()
                times.append(time.perf_counter() - start)
                values = torch.cat(outputs)
                require(torch.isfinite(values).all().item() and len(values) == len(dataset), 'Invalid benchmark outputs')
                checksums.append(float(values.double().sum()))
            peak = memory()
        return duration_summary(times, len(dataset)), peak, checksums

    try:
        cache = control.ContextCache(options.model_dir, options.context_cache, device, 16)
        features, context = {}, {}
        for split in ('train', 'dev'):
            clear()
            sync()
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            context[split], cold = cache.get(task, split, case.rows[split])
            sync()
            seconds, peak = time.perf_counter() - start, memory()
            require(not cold['reused'], 'Fresh extraction required,never silently reuse a prior timing attempt')
            reloaded, warm = cache.get(task, split, case.rows[split])
            require(np.array_equal(reloaded, context[split]), 'Warm payload differs')
            features[split] = dict(cold=cold, synchronized_wall_seconds=seconds, memory=peak, warm=warm,
                                   rows=len(case.rows[split]))
            del reloaded
        clear()
        # Measure the original teacher-logit procedure with the actual selected checkpoint.
        sync()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        logits, logit_manifest = teacher.teacher_logits(case, Path(dependencies['teacher_view']), directory / 'fresh_teacher_logits')
        sync()
        logit_cost = dict(rows=len(logits), freshly_extracted=True, seconds=time.perf_counter()-start,
            manifest=logit_manifest, memory=memory())
        require(logit_manifest['identity']['teacher'] == dependencies['teachers'][task]['provenance'], 'Teacher differs')
        clear()
        teacher_model = case.model('dnabert_finetune', 42).to(device)
        loader = fair.make_loader(case, 'dnabert_finetune', 'train', 42, 4, True)
        teacher_epoch, teacher_peak = measure_epoch(teacher_model, loader, micro=4)
        del teacher_model, loader
        clear()
        length = 1000 if task == 'H3K27me3' else 600
        datasets, preparation = {}, {}
        for variant in ('local_only', 'local_comp', 'full'):
            start = time.perf_counter()
            datasets[variant] = {split: control.tensors(rows, length,
                context[split] if variant == 'full' else np.zeros((len(rows), 1), dtype='float32'))
                for split, rows in case.rows.items()}
            preparation[variant] = dict(seconds=time.perf_counter()-start,
                rows=30000, scope='CPU sequence encoding and composition preprocessing;train+dev')
        records = []
        for seed in SEEDS:
            for variant, teacher_mode in case_order(seed):
                clear()
                control.seed_everything(seed)
                model = control.build_model(variant, length, case.classes, 128, [3, 5, 7, 11, 15]).to(device)
                train_data = datasets[variant]['train']
                if teacher_mode:
                    train_data = TensorDataset(*train_data.tensors, torch.from_numpy(logits))
                loader = DataLoader(train_data, batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(seed), num_workers=0)
                training, train_peak = measure_epoch(model, loader, teacher_mode=teacher_mode)
                del loader
                model.zero_grad(set_to_none=True)
                clear()
                inference, infer_peak, checksums = infer(model, datasets[variant]['dev'], 16, 3)
                latency_loader = DataLoader(datasets[variant]['dev'], batch_size=1, shuffle=False)
                latency = []
                with torch.inference_mode():
                    for batch in itertools.islice(iter(latency_loader), 64):
                        sync()
                        start = time.perf_counter()
                        bases, comp, pooled, _ = batch
                        with control.amp_context(device, True):
                            output = model({}, bases.to(device), comp.to(device), bert_pooled=pooled.to(device)).float().cpu()
                        sync()
                        latency.append(time.perf_counter() - start)
                        require(torch.isfinite(output).all().item(), 'Nonfinite latency output')
                entry = dict(seed=seed, variant=variant, teacher=teacher_mode,
                    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    training=training, training_memory=train_peak, inference=inference,
                    inference_memory=infer_peak, latency_seconds=latency,
                    latency_median_seconds=statistics.median(latency), output_checksums=checksums)
                records.append(entry)
                control.save_json(directory / 'measurement_progress.json', dict(task=task, cases=records, accuracy_results=0))
                print('BENCHMARK', task, seed, variant, teacher_mode, training['seconds'], flush=True)
                del model, output, train_data
                clear()
        stopped.set()
        monitor_thread.join(timeout=25)
        require(not monitor_thread.is_alive(), 'Resource monitor did not terminate')
        result = dict(task=task, protocol=protocol(), cases=records, features=features, preprocessing=preparation,
            teacher_epoch=teacher_epoch, teacher_epoch_memory=teacher_peak, teacher_logits=logit_cost,
            original_teacher_observed_costs=dependencies['teachers'][task],
            monitor_samples=samples, monitor_errors=monitor_errors,
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device), python=sys.version.split()[0]),
            accuracy_results=0, test_results=0, timings_are_full_training_runs=False,
            isolated_from_own_accuracy_training=True,
            foreign_compute_presence_requires_qualified_not_unconditional_speed_claims=True)
        validate_task_result(result, task)
        control.save_json(directory / 'benchmark.json', result)
        return result
    finally:
        stopped.set()
        monitor_thread.join(timeout=25)
