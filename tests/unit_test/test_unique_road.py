'''
test select beam.
'''
import random
import torch
from core.extensions import beam_search_unique_roads


def _python_unique_road(scores, roads, beam_size, recombine_sum):
    '''python unique road'''
    scores, roads = scores.cpu(), roads.cpu()
    bsz, expand_beam_size = scores.shape
    roads_lens = []
    roads_list = [
        [tuple(item for item in beam if item) for beam in batch] for batch in roads.tolist()
    ]
    for bid in range(bsz):
        roads_batch, roads_len = roads_list[bid], []
        compose_dict = dict()
        for beam_idx in range(beam_size):
            road = roads_batch[beam_idx]
            compose_dict[road] = beam_idx
            roads_len.append(len(road))
        scores_beams, roads_beams = scores[bid], roads[bid]
        for beam_idx in range(beam_size, expand_beam_size):
            road = roads_batch[beam_idx]
            beam_idx_ = compose_dict.get(road)
            if beam_idx_ is not None:
                scores_beam_idx_value = scores_beams[beam_idx]
                if recombine_sum:
                    scores_beams[beam_idx_] = torch.logsumexp(
                        torch.tensor([scores_beams[beam_idx_], scores_beam_idx_value]), 0
                    )
                elif scores_beams[beam_idx_] < scores_beam_idx_value:
                    scores_beams[beam_idx_] = scores_beam_idx_value
                    roads_beams[beam_idx_] = roads_beams[beam_idx]
                scores_beams[beam_idx] = -1e8
            else:
                compose_dict[road] = beam_idx
            roads_len.append(len(road))
        roads_lens.append(roads_len)
    scores, roads = scores.cuda(), roads.cuda()
    roads_lens = torch.tensor(roads_lens, dtype=scores.dtype, device='cuda')
    return scores, roads, roads_lens


def _generate_dataset(batch_size, beam_size, expand_num, road_len):
    '''generate dataset'''
    concate_scores = torch.rand(
        batch_size, (expand_num + 1) * beam_size, dtype=torch.float, device='cuda'
    )
    concate_hyp_tgt = torch.randint(
        0,
        12345,
        (batch_size, (expand_num + 1) * beam_size, road_len),
        dtype=torch.int64,
        device='cuda',
    )
    for batch_id in range(batch_size):
        for neam_id in range(beam_size, (expand_num + 1) * beam_size):
            if random.random() > 0.3:
                neam_id_ = random.randint(0, neam_id - 1)
                concate_hyp_tgt[batch_id, neam_id, :] = concate_hyp_tgt[batch_id, neam_id_, :]
    return concate_scores, concate_hyp_tgt


def test_unique_road():
    '''test beam select'''
    batch_size, beam_size, expand_num, road_len = 193, 10, 2, 100
    scores, roads = _generate_dataset(batch_size, beam_size, expand_num, road_len)
    for retain_align_info in [False, True]:
        for recombine_sum in [False, True]:
            py_scores, py_roads, py_lens = _python_unique_road(
                scores, roads, beam_size, recombine_sum
            )
            cpp_scores, cpp_roads, cpp_lens = beam_search_unique_roads(
                scores, roads, beam_size, recombine_sum, retain_align_info
            )
            assert torch.allclose(py_scores, cpp_scores)
            assert torch.allclose(py_lens, cpp_lens)
            if retain_align_info:
                assert torch.allclose(py_roads, cpp_roads)
