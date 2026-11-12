"""CUDA graph backends for adaptive drafting with reusable static KV buffers.

GroupedDraftGraphs is the default: ordinary graphs share one memory pool, with
host routing between exit groups. ConditionalDraftGraphs is an experimental
opt-in requiring PyTorch's conditional-node API. Its branch streams can reserve
large workspaces and affected PyTorch builds leak its pools after destruction.
"""

import torch


def conditional_graphs_available(model):
    return (
        model.model.embed_tokens.weight.device.type == "cuda"
        and hasattr(torch.cuda.CUDAGraph, "begin_capture_to_if_node")
        and hasattr(torch.cuda.CUDAGraph, "end_capture_to_conditional_node")
    )


class _GraphKVCache:
    """The decoder's update protocol, writing into persistent graph inputs."""

    def __init__(self, model, capacity):
        parameter = model.model.embed_tokens.weight
        config = model.config
        shape = (1, config.num_key_value_heads, capacity, config.hidden_size // config.num_attention_heads)
        self.key_cache = [parameter.new_zeros(shape) for _ in model.model.layers]
        self.value_cache = [parameter.new_zeros(shape) for _ in model.model.layers]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        positions = cache_kwargs["cache_position"]
        keys, values = self.key_cache[layer_idx], self.value_cache[layer_idx]
        keys.index_copy_(2, positions, key_states)
        values.index_copy_(2, positions, value_states)
        return keys, values


class ConditionalDraftGraphs:
    """Capture one draft/verify graph per possible draft-block length.

    Cache positions and per-layer completion lengths are device tensors. A
    layer's catch-up pass has one conditional branch for each possible missing
    suffix length; exactly one runs. This preserves KV/state reuse when exit
    depth changes without reading completion lengths back to Python.

    The verified prefix is prefilled once. At a new round, resetting the device
    base position performs logical rollback: stale positions are masked and
    overwritten. No cached prefix tensors are copied on each decoding step.
    """

    @torch.inference_mode()
    def __init__(self, model, classifier, checks, threshold, capacity, max_drafts):
        if not conditional_graphs_available(model):
            raise RuntimeError("This PyTorch/CUDA build has no CUDA conditional graph API.")
        self.model, self.classifier = model, classifier
        self.checks, self.threshold = checks, threshold
        self.capacity, self.max_drafts = capacity, max_drafts
        parameter = model.model.embed_tokens.weight
        self.device = parameter.device
        self.cache = _GraphKVCache(model, capacity)
        width = max_drafts + 1
        hidden = model.config.hidden_size
        self.states = [parameter.new_zeros((1, width, hidden)) for _ in range(len(model.model.layers) + 1)]
        self.token_ids = torch.zeros((1, width), device=self.device, dtype=torch.long)
        self.base = torch.zeros((), device=self.device, dtype=torch.long)
        self.completed = torch.zeros(len(model.model.layers), device=self.device, dtype=torch.long)
        self.active = torch.ones((), device=self.device, dtype=torch.bool)
        self.invalid_score = torch.zeros((), device=self.device, dtype=torch.bool)
        self.exit_depth = torch.zeros((), device=self.device, dtype=torch.long)
        self.gru_state = parameter.new_zeros((classifier.config["num_layers"], 1, classifier.config["hidden_size"]))
        self.selected_raw = parameter.new_zeros((1, 1, hidden))
        self.draft_output = parameter.new_zeros((1, model.config.vocab_size))
        self.verify_output = parameter.new_zeros((1, width, model.config.vocab_size))
        self.key_positions = torch.arange(capacity, device=self.device)
        self.draft_graphs, self.verify_graphs = {}, {}
        self.length = 1

        # cuBLAS/cuDNN must be initialized before capture, including every GRU
        # group shape. Graph construction/warmup is outside the generation timer.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            for width in range(1, max_drafts + 2):
                self.completed.zero_()
                self._execute(width, draft=True, graph=None)
                self.completed.zero_()
                self._execute(width, draft=False, graph=None)
            stream.synchronize()
            for width in range(1, max_drafts + 2):
                if width <= max_drafts:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self._execute(width, draft=True, graph=graph)
                    self.draft_graphs[width] = graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self._execute(width, draft=False, graph=graph)
                self.verify_graphs[width] = graph
        torch.cuda.current_stream(self.device).wait_stream(stream)

    def _position_data(self, width):
        data = {}
        for missing in range(1, width + 1):
            positions = self.base + torch.arange(width - missing, width, device=self.device)
            blocked = self.key_positions[None, :] > positions[:, None]
            mask = self.selected_raw.new_zeros((1, 1, missing, self.capacity))
            mask.masked_fill_(blocked[None, None], torch.finfo(mask.dtype).min)
            position_ids = positions.unsqueeze(0)
            rotary = self.model.model.rotary_emb(self.states[0][:, :missing], position_ids)
            data[missing] = positions, position_ids, mask, rotary
        return data

    def _layer(self, index, width, position_data, graph):
        # During capture every possible body is recorded, not executed. On
        # replay the device completion counter selects only the missing suffix.
        for missing, (positions, position_ids, mask, rotary) in position_data.items():
            if graph is not None:
                graph.begin_capture_to_if_node(self.completed[index] == width - missing)
            hidden = self.states[index][:, width - missing:width]
            result = self.model.model.layers[index](
                hidden, attention_mask=mask, position_ids=position_ids,
                past_key_value=self.cache, use_cache=True, output_attentions=False,
                cache_position=positions, position_embeddings=rotary,
            )
            raw = result[0] if isinstance(result, tuple) else result
            self.states[index + 1][:, width - missing:width].copy_(raw)
            self.completed[index].fill_(width)
            if graph is not None:
                graph.end_capture_to_conditional_node()

    def _execute(self, width, draft, graph):
        self.states[0][:, :width].copy_(self.model.model.embed_tokens(self.token_ids[:, :width]))
        positions = self._position_data(width)
        if draft:
            self.active.fill_(True)
            self.gru_state.zero_()
            start = 0
            for depth in self.checks:
                if graph is not None:
                    graph.begin_capture_to_if_node(self.active)
                for index in range(start, depth):
                    self._layer(index, width, positions, graph)
                raw = self.states[depth][:, width - 1:width]
                self.selected_raw.copy_(raw)
                self.exit_depth.fill_(depth)
                if depth != self.checks[-1]:
                    inputs = torch.cat([
                        self.states[index + 1][:, width - 1:width] for index in range(start, depth)
                    ], dim=1)
                    states, state = self.classifier.gru(inputs, self.gru_state)
                    self.gru_state.copy_(state)
                    score = self.classifier.head(states[:, -1]).sigmoid().reshape(())
                    finite = torch.isfinite(score)
                    self.invalid_score.logical_or_(~finite)
                    self.active.copy_(finite & (score < self.threshold))
                else:
                    self.active.fill_(False)
                if graph is not None:
                    graph.end_capture_to_conditional_node()
                start = depth
            self.draft_output.copy_(self.model.lm_head(self.model.model.norm(self.selected_raw))[:, 0])
        else:
            for index in range(len(self.model.model.layers)):
                self._layer(index, width, positions, graph)
            raw = self.states[-1][:, :width]
            self.verify_output[:, :width].copy_(self.model.lm_head(self.model.model.norm(raw)))

    @torch.inference_mode()
    def prefill(self, input_ids):
        """Initialize all layers' verified prefix, excluding its final token."""
        prefix_length = input_ids.shape[1] - 1
        if prefix_length:
            result = self.model.model(input_ids[:, :-1], use_cache=True)
            for index in range(len(self.model.model.layers)):
                keys, values = result.past_key_values[index]
                self.cache.key_cache[index][:, :, :prefix_length].copy_(keys)
                self.cache.value_cache[index][:, :, :prefix_length].copy_(values)

    def begin_round(self, context):
        self.length = 1
        self.base.fill_(context.shape[1] - 1)
        self.completed.zero_()
        self.invalid_score.zero_()
        self.token_ids[:, :1].copy_(context[:, -1:])
        return self

    def append(self, token):
        self.token_ids[:, self.length].fill_(token)
        self.length += 1

    def draft_logits(self, classifier, threshold, max_depth, initial_fraction=0.25):
        self.draft_graphs[self.length].replay()
        # Clone because the next graph replay reuses this output buffer.
        return self.draft_output.clone(), self.exit_depth

    def verification_logits(self, count):
        if count != self.length:
            raise ValueError("Verification length must match the draft suffix.")
        self.verify_graphs[self.length].replay()
        # One error check per round, never a host decision at an exit checkpoint.
        if self.invalid_score.item():
            raise ValueError("The early-exit classifier produced a non-finite score.")
        return self.verify_output[:, :count].clone()


class GroupedDraftGraphs:
    """Ordinary CUDA graphs with one host exit decision per decoder group.

    All captures share one pool and one stream. Graph intermediates never
    escape: outputs are copied into buffers allocated before capture. This
    permits arbitrary sequential replay order without conditional-node streams
    or their per-stream BLAS workspaces. Completion lengths live on the host;
    each decoder group runs only its missing suffix, including on verification.
    Instances are single-request/single-stream, like ConditionalDraftGraphs.
    """

    @torch.inference_mode()
    def __init__(self, model, classifier, checks, threshold, capacity, max_drafts):
        if classifier is None and len(checks) != 1:
            raise ValueError("Fixed-depth graphs require exactly one exit depth.")
        self.model, self.classifier = model, classifier
        self.checks, self.threshold = checks, threshold
        self.capacity, self.max_drafts = capacity, max_drafts
        parameter = model.model.embed_tokens.weight
        self.device = parameter.device
        self.cache = _GraphKVCache(model, capacity)
        width = max_drafts + 1
        hidden = model.config.hidden_size
        # Only group boundaries are needed between calls. Intermediate layer
        # states used by the GRU have graph-local lifetimes.
        boundaries = [0, *checks]
        if boundaries[-1] != len(model.model.layers):
            boundaries.append(len(model.model.layers))
        self.groups = list(zip(boundaries, boundaries[1:]))
        self.states = {depth: parameter.new_zeros((1, width, hidden)) for depth in boundaries}
        self.hidden_input = parameter.new_zeros((1, width, hidden))
        self.hidden_output = parameter.new_zeros((1, width, hidden))
        self.position_start = torch.zeros((), device=self.device, dtype=torch.long)
        self.key_positions = torch.arange(capacity, device=self.device)
        self.gru_state = (
            parameter.new_zeros((classifier.config["num_layers"], 1, classifier.config["hidden_size"]))
            if classifier is not None else None
        )
        self.score = parameter.new_zeros(())
        self.draft_output = parameter.new_zeros((1, model.config.vocab_size))
        self.verify_output = parameter.new_zeros((1, width, model.config.vocab_size))
        self.draft_graphs, self.verify_graphs, self.head_graphs = {}, {}, {}
        self.completed = {start: 0 for start, _ in self.groups}
        self.length, self.base = 1, 0
        self.pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            # Warm every shape before capture to initialize CUDA libraries.
            for missing in range(1, width + 1):
                for start, end in self.groups:
                    self._execute_group(start, end, missing, draft=False)
                    if end <= checks[-1] and missing <= max_drafts:
                        self._execute_group(start, end, missing, draft=True)
                self._execute_head(missing)
            stream.synchronize()
            for missing in range(1, width + 1):
                for start, end in self.groups:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                        self._execute_group(start, end, missing, draft=False)
                    self.verify_graphs[start, missing] = graph
                    if end <= checks[-1] and missing <= max_drafts:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                            self._execute_group(start, end, missing, draft=True)
                        self.draft_graphs[start, missing] = graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool, stream=stream):
                    self._execute_head(missing)
                self.head_graphs[missing] = graph
        torch.cuda.current_stream(self.device).wait_stream(stream)

    def _execute_group(self, start, end, missing, draft):
        hidden = self.hidden_input[:, :missing]
        positions = self.position_start + torch.arange(missing, device=self.device)
        position_ids = positions.unsqueeze(0)
        mask = hidden.new_zeros((1, 1, missing, self.capacity))
        mask.masked_fill_(self.key_positions[None, :] > positions[:, None], torch.finfo(hidden.dtype).min)
        rotary = self.model.model.rotary_emb(hidden, position_ids)
        classify = draft and end != self.checks[-1]
        raw_states = []
        for index in range(start, end):
            result = self.model.model.layers[index](
                hidden, attention_mask=mask, position_ids=position_ids,
                past_key_value=self.cache, use_cache=True, output_attentions=False,
                cache_position=positions, position_embeddings=rotary,
            )
            hidden = result[0] if isinstance(result, tuple) else result
            if classify:
                raw_states.append(hidden[:, -1:])
        self.hidden_output[:, :missing].copy_(hidden)
        if classify:
            states, state = self.classifier.gru(torch.cat(raw_states, dim=1), self.gru_state)
            self.gru_state.copy_(state)
            self.score.copy_(self.classifier.head(states[:, -1]).sigmoid().reshape(()))

    def _execute_head(self, count):
        logits = self.model.lm_head(self.model.model.norm(self.hidden_input[:, :count]))
        self.verify_output[:, :count].copy_(logits)
        if count == 1:
            self.draft_output.copy_(logits[:, 0])

    def _run_group(self, start, end, draft):
        completed = self.completed[start]
        missing = self.length - completed
        if not missing:
            return
        self.hidden_input[:, :missing].copy_(self.states[start][:, completed:self.length])
        self.position_start.fill_(self.base + completed)
        graphs = self.draft_graphs if draft else self.verify_graphs
        graphs[start, missing].replay()
        self.states[end][:, completed:self.length].copy_(self.hidden_output[:, :missing])
        self.completed[start] = self.length

    @torch.inference_mode()
    def prefill(self, input_ids):
        prefix_length = input_ids.shape[1] - 1
        if prefix_length:
            result = self.model.model(input_ids[:, :-1], use_cache=True)
            for index in range(len(self.model.model.layers)):
                keys, values = result.past_key_values[index]
                self.cache.key_cache[index][:, :, :prefix_length].copy_(keys)
                self.cache.value_cache[index][:, :, :prefix_length].copy_(values)

    def begin_round(self, context):
        self.length, self.base = 1, context.shape[1] - 1
        for start in self.completed:
            self.completed[start] = 0
        self.states[0][:, :1].copy_(self.model.model.embed_tokens(context[:, -1:]))
        return self

    def append(self, token):
        ids = torch.tensor([[token]], device=self.device)
        self.states[0][:, self.length:self.length + 1].copy_(self.model.model.embed_tokens(ids))
        self.length += 1

    def draft_logits(self, classifier, threshold, max_depth, initial_fraction=0.25):
        if self.gru_state is not None:
            self.gru_state.zero_()
        for start, end in self.groups:
            self._run_group(start, end, draft=True)
            if end == self.checks[-1]:
                break
            score = self.score.item()
            if not 0 <= score <= 1:
                raise ValueError("The early-exit classifier produced a non-finite score.")
            if score >= self.threshold:
                break
        self.hidden_input[:, :1].copy_(self.states[end][:, self.length - 1:self.length])
        self.head_graphs[1].replay()
        return self.draft_output.clone(), end

    def verification_logits(self, count):
        if count != self.length:
            raise ValueError("Verification length must match the draft suffix.")
        for start, end in self.groups:
            self._run_group(start, end, draft=False)
        self.hidden_input[:, :count].copy_(self.states[end][:, :count])
        self.head_graphs[count].replay()
        return self.verify_output[:, :count].clone()
