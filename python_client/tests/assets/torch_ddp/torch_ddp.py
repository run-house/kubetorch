def torch_ddp(epochs):
    import torch

    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.distributed.init_process_group(backend="gloo")
    rank = torch.distributed.get_rank()
    print(f"Rank {rank} of {torch.distributed.get_world_size()} initialized")

    model = torch.nn.Linear(10, 1)
    model = DDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a simple training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(torch.randn(10))
        loss = output.sum()
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}: Epoch {epoch}, Loss {loss.item()}")

    torch.distributed.destroy_process_group()
    return "Success"
