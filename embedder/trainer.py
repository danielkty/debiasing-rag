import torch
import transformers
from tqdm import trange
from accelerate import Accelerator

from loss import SimCSE

class EmbeddingTrainer:
    def __init__(self, model):
        self.model = model

    def fit(
            self, 
            *,
            train_loader=None, 
            epochs=None,
            optimizer=None, 
            optimizer_args=None,
            criterion=None, 
            criterion_args=None,
            scheduler=None,
            scheduler_args=None,
            grad_clip=None,
            chunk_size=None,
            accelerator=None,
            show_progress_bar=False,
            use_wandb=False,):
        
        # use_gradcache = chunk_size is not None
        use_gradcache = chunk_size != 0

        if accelerator is None:
            accelerator = Accelerator()

        if use_wandb:
            import wandb

        steps_per_epoch = len(train_loader)


        # Set up loss
        if criterion == 'simcse':
            train_loss = SimCSE(self.model, **criterion_args)
        else:
            raise ValueError(f'Invalid criterion: {criterion}')


        # Set up optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        named_parameters = list(train_loss.model.named_parameters()) if use_gradcache else list(train_loss.named_parameters())
        weight_decay = optimizer_args.pop('weight_decay', None)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay if weight_decay is not None else 0.0},
            {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_args)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, **optimizer_args)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, **optimizer_args)
        else:
            raise ValueError(f'Invalid optimizer: {optimizer}')


        # Set up scheduler
        if scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
        elif scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler == 'warmup_linear':
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer, **scheduler_args)
        elif scheduler == 'constant':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1)
        else:
            raise ValueError(f'Invalid scheduler: {scheduler}')

        
        # wrap with accelerator and prepare gradcache; note: train_loader is handled manually
        if use_gradcache:
            train_loss.model, optimizer, scheduler = accelerator.prepare(
                train_loss.model, optimizer, scheduler)
            train_loss.models = [train_loss.model for _ in range(len(train_loss.models))]
        else:
            train_loss, optimizer, scheduler = accelerator.prepare(
                train_loss, optimizer, scheduler)

        # Train
        show_progress_bar = accelerator.is_main_process and show_progress_bar
        data_iterator = iter(train_loader)
        epoch_iterator = trange(epochs, desc='Epoch', disable=not show_progress_bar)
        for epoch in epoch_iterator:

            # zero the parameter gradients
            if use_gradcache:
                train_loss.model.zero_grad()
                train_loss.model.train()
            else:
                train_loss.zero_grad()
                train_loss.train()

            # Train for an epoch
            n_iters = steps_per_epoch
            batch_iterator = trange(n_iters, desc='loss=', disable=not show_progress_bar)
            for _ in batch_iterator:
                # get the next batch
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_loader)
                    data = next(data_iterator)

                # manually segment data by process
                if accelerator is not None:
                    batch_size = len(data[0])
                    device_batch_size = batch_size // accelerator.num_processes
                    pid = accelerator.process_index
                    data = [x[pid*device_batch_size:(pid+1)*device_batch_size] for x in data]

                # forward pass
                loss_value = train_loss(data)

                # backward pass
                if use_gradcache:
                    # gradcache will handle the backward pass
                    torch.nn.utils.clip_grad_norm_(train_loss.model.parameters(), grad_clip)
                else:
                    accelerator.backward(loss_value)
                    torch.nn.utils.clip_grad_norm_(train_loss.parameters(), grad_clip)

                # optimizer step
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # update progress bar
                batch_iterator.set_description(f'loss={loss_value.item():.4f}')
                # wandb
                if use_wandb:
                    wandb.log({'loss': loss_value.item()})
