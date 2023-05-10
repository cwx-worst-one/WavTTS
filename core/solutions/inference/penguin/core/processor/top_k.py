import numpy


class TopK:
    def __init__(self, dims=(32,)):
        self.t_hyps_scores = numpy.zeros(dims)
        self.u_hyps_scores = numpy.zeros(dims)
        self.u_hyps_topk_idx = numpy.zeros(dims)

    def process(self, message, beam):
        dim = message.hyp_scores.shape
        self.t_hyps_scores = numpy.zeros((dim))
        self.u_hyps_scores = numpy.zeros((dim))
        self.u_hyps_topk_idx = numpy.zeros((dim))
        self.cal_topk(message.jointer_output, message.hyp_scores, beam)

    def cal_topk(self, jointer_output, hyp_scores, beam):
        batch_size = hyp_scores.shape[0]
        new_hyp_scores = jointer_output + numpy.expand_dims(hyp_scores, -1)
        self.t_hyps_scores = jointer_output[:, :, 0] + hyp_scores
        new_hyp_scores[:, :, 0] = -100000.0
        new_hyp_scores = numpy.reshape(new_hyp_scores, (batch_size, -1))
        topk_index = numpy.argpartition(new_hyp_scores, -beam, axis=-1)[:, -beam:]
        self.u_hyps_scores = numpy.array(
            [list(new_hyp_scores[i, topk_index[i, :]]) for i in range(batch_size)]
        )
        self.u_hyps_topk_idx = topk_index
