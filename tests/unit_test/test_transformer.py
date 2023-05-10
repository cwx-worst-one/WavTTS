''' Tensor Parallel Transformer test. '''
import torch
import torch.nn.functional as F
from core.utils import (
    logging,
    get_rank,
    distributed_init,
    get_world_size,
    dist_broadcast_model,
    dist_broadcast,
)
from core.models.layers.transformer import BiTransformerLayer, ParallelTransformer


def _test_transformer_speed(tp_size):
    layer_result_norm = 'instance'  # or ''
    for bsz, seq in [
        (64, 512),
        (32, 512),
        # 48K
        (24, 498),
        (24, 500),
        (26, 442),
        (26, 448),
        (26, 450),
        (30, 398),
        (34, 348),
        # 24K
        (12, 494),
        (12, 498),
        (12, 500),
        (13, 436),
        (13, 442),
        (13, 450),
        (15, 394),
        (15, 398),
        (17, 348),
    ]:
        emb, head, ffn, act = 4096, 32, 16384, 'gelu'
        m = ParallelTransformer(
            emb,
            head,
            ffn,
            0.2,
            0.2,
            activation=act,
            tensor_parallel=tp_size,
        )
        m.cuda()
        m.train()

        x = torch.rand([bsz, seq, emb], device='cuda')
        st, ed = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        with torch.cuda.nvtx.range(
            'bsz {} seq {} emb {} head {} ffn {} tp {}'.format(bsz, seq, emb, head, ffn, tp_size)
        ):
            torch.set_autocast_enabled(True)
            for _ in range(10):
                with torch.cuda.nvtx.range('fwd'):
                    y, _, layer_result = m(
                        x,
                        fused=True,
                        batch_first=True,
                        output_layer_result=True,
                        layer_result_norm=layer_result_norm,
                    )
                with torch.cuda.nvtx.range('bwd'):
                    torch.autograd.backward(
                        [y, layer_result],
                        grad_tensors=[y, layer_result],
                    )
            st.record()
            for _ in range(10):
                with torch.cuda.nvtx.range('fwd'):
                    y, _, layer_result = m(
                        x,
                        fused=True,
                        batch_first=True,
                        output_layer_result=True,
                        layer_result_norm=layer_result_norm,
                    )
                with torch.cuda.nvtx.range('bwd'):
                    torch.autograd.backward(
                        [y, layer_result],
                        grad_tensors=[y, layer_result],
                    )
            ed.record()
            ed.synchronize()
            ms = st.elapsed_time(ed) / 10
            torch.set_autocast_enabled(False)
        flop = bsz * seq * 3 * emb * emb * 2  # in proj
        flop += bsz * head * seq * seq * (emb // head) * 2  # bmm(q, k)
        flop += bsz * head * seq * (emb // head) * seq * 2  # bmm(w, v)
        flop += bsz * seq * emb * emb * 2  # out proj
        flop += bsz * seq * ffn * emb * 2  # fc1
        flop += bsz * seq * ffn * emb * 2  # fc2
        flop *= 3  # forward and backward
        flop /= tp_size
        tflops = flop / ms / 1e9

        if get_rank() != 0:
            continue
        print()
        print('bsz', bsz, 'seq', seq, 'emb', emb, 'head', head, 'ffn', ffn, 'tp', tp_size)
        print('flop: ', flop / 1e12, 'TFlop')
        print('time: ', ms, 'ms')
        print('flops: ', tflops, 'TFlops')
        print('flops percent: ', tflops / 311.86944 * 100, '%')


def _test_tp_transformer(tp_size):
    """test tensor parallel transformer in multi cards"""
    bsz, seq = 37, 800
    emb, head, ffn, act = 4096, 32, 16384, 'gelu'
    layer_result_norm = 'instance'
    m1 = BiTransformerLayer(emb, head, ffn, 0, 0, activation=act)
    m1.cuda()
    m1.train()

    dist_broadcast_model(m1, root_rank=0)
    x1 = (torch.rand([bsz, seq, emb], device='cuda') - 0.5).detach().requires_grad_()
    dy = torch.rand([bsz, seq, emb], device='cuda') - 0.5
    dlr = torch.rand([bsz, seq, emb], device='cuda') - 0.5

    dist_broadcast(x1, 0)
    dist_broadcast(dy, 0)
    dist_broadcast(dlr, 0)

    torch.set_autocast_enabled(True)
    y1, _, layer_result1 = m1(x1, fused=True, batch_first=True, output_layer_result=True)
    if layer_result_norm == 'instance':
        t1 = layer_result1.permute(0, 2, 1)  # [bsz, seq, emb] -> [bsz, emb, seq]
        t3 = F.instance_norm(t1.float())
        layer_result1 = t3.permute(0, 2, 1)  # [bsz, emb, seq] -> [bsz, seq, emb]
    torch.autograd.backward(
        [y1, layer_result1],
        grad_tensors=[
            dy.to(y1.dtype),
            dlr.to(layer_result1.dtype),
        ],
    )
    torch.set_autocast_enabled(False)

    x2 = x1.clone().detach().requires_grad_()
    m2 = ParallelTransformer(emb, head, ffn, 0, 0, activation=act, tensor_parallel=tp_size)
    m2.cuda()
    m2.train()
    m2.load_state_dict(m1.state_dict())

    torch.set_autocast_enabled(True)
    y2, _, layer_result2 = m2(
        x2,
        fused=True,
        batch_first=True,
        output_layer_result=True,
        layer_result_norm=layer_result_norm,
    )
    torch.autograd.backward(
        [y2, layer_result2],
        grad_tensors=[
            dy.to(y2.dtype),
            dlr.to(layer_result2.dtype),
        ],
    )
    torch.set_autocast_enabled(False)

    logging.error('rank %d tp_size %d world_size %d', get_rank(), tp_size, get_world_size())
    logging.error('rank %d x %r', get_rank(), (x1 - x2).abs().max())
    logging.error('rank %d y %r', get_rank(), (y1 - y2).abs().max())
    atol = (layer_result1 - layer_result2).abs().max()
    logging.error('rank %d layer_result %r', get_rank(), atol)
    atol = (x1.grad - x2.grad).abs().max()
    logging.error('rank %d dx %r', get_rank(), atol)
    atol = (m2.self_attn_layer_norm.weight.grad - m1.self_attn_layer_norm.weight.grad).abs().max()
    logging.error('rank %d self_attn_layer_norm_weight %r', get_rank(), atol)
    atol = (m2.self_attn_layer_norm.bias.grad - m1.self_attn_layer_norm.bias.grad).abs().max()
    logging.error('rank %d self_attn_layer_norm_bias %r', get_rank(), atol)
    local_head = head // tp_size
    local_rank = get_rank() % tp_size
    st, ed = local_head * local_rank, local_head * (local_rank + 1)
    v1 = m2.self_attn_in_proj_weight.grad.reshape(3, local_head, -1, emb)
    v2 = m1.self_attn.in_proj.weight.grad.reshape(3, head, -1, emb)
    atol = (v2[:, st:ed, :, :] - v1).abs().max()
    logging.error('rank %d self_attn_in_proj_weight %r', get_rank(), atol)
    v1 = m2.self_attn_in_proj_bias.grad.reshape(3, local_head, -1)
    v2 = m1.self_attn.in_proj.bias.grad.reshape(3, head, -1)
    atol = (v2[:, st:ed, :] - v1).abs().max()
    logging.error('rank %d self_attn.in_proj.bias %r', get_rank(), atol)
    v1 = m2.self_attn_out_proj_weight.grad.reshape(emb, local_head, -1)
    v2 = m1.self_attn.out_proj.weight.grad.reshape(emb, head, -1)
    atol = (v2[:, st:ed, :] - v1).abs().max()
    logging.error('rank %d self_attn.out_proj.weight %r', get_rank(), atol)
    atol = (m1.self_attn.out_proj.bias.grad - m2.self_attn_out_proj_bias.grad).abs().max()
    logging.error('rank %d self_attn.out_proj.bias %r', get_rank(), atol)
    atol = (m1.final_layer_norm.weight.grad - m2.final_layer_norm.weight.grad).abs().max()
    logging.error('rank %d final_layer_norm.weight %r', get_rank(), atol)
    atol = (m1.final_layer_norm.bias.grad - m2.final_layer_norm.bias.grad).abs().max()
    logging.error('rank %d final_layer_norm.bias %r', get_rank(), atol)
    local_ffn = ffn // tp_size
    st, ed = local_rank * local_ffn, local_ffn * (local_rank + 1)
    v1 = m1.fc1.weight.grad
    v2 = m2.fc1_weight.grad
    atol = (v1[st:ed, :] - v2).abs().max()
    logging.error('rank %d fc1.weight %r', get_rank(), atol)
    v1 = m1.fc1.bias.grad
    v2 = m2.fc1_bias.grad
    atol = (v1[st:ed] - v2).abs().max()
    logging.error('rank %d fc1.bias %r', get_rank(), atol)
    v1 = m1.fc2.weight.grad
    v2 = m2.fc2_weight.grad
    atol = (v1[:, st:ed] - v2).abs().max()
    logging.error('rank %d fc2.weight %r', get_rank(), atol)
    atol = (m1.fc2.bias.grad - m2.fc2_bias.grad).abs().max()
    logging.error('rank %d fc2.bias %r', get_rank(), atol)


def _test_python_nccl():
    distributed_init()
    tp_size = 4
    rank = get_rank()

    st0 = torch.cuda.current_stream()
    st1 = torch.cuda.Stream()
    evt0 = torch.cuda.Event()
    evt1 = torch.cuda.Event()

    if rank == 0:
        uids = [torch.cuda.nccl.unique_id() for _ in range(get_world_size() // tp_size)]
    else:
        uids = [b'' for _ in range(get_world_size() // tp_size)]
    torch.distributed.broadcast_object_list(uids, src=0)
    uid = uids[rank // tp_size]
    logging.error('rank %d tp_size %d uid %r', rank, tp_size, uid)
    comm = torch.cuda.nccl.init_rank(int(tp_size), uid, int(rank % tp_size))

    for i in range(5):
        with torch.cuda.nvtx.range('{}'.format(i)):
            x1 = torch.ones([37, 400, 4096], device='cuda')
            x2 = x1 * rank
            x3 = torch.zeros_like(x2)
            st0.record_event(evt0)
            st1.wait_event(evt0)
            torch.cuda.nccl.all_reduce([x2], [x3], 0, [st1], [comm])
            st1.record_event(evt1)
            x4 = torch.nn.functional.gelu(x2)
            st0.wait_event(evt1)
            x5 = x4 + x3
            logging.error(
                'rank %d x2 %r x3 %r x4 %r x5 %r',
                rank,
                x2[0, 0, :4],
                x3[0, 0, :4],
                x4[0, 0, :4],
                x5[0, 0, :4],
            )


if __name__ == '__main__':
    torch.manual_seed(220315)
    torch.cuda.manual_seed(2113)
    # init distributed env
    distributed_init()
    tp_size = get_world_size()
    _test_tp_transformer(tp_size)
    _test_transformer_speed(tp_size)
