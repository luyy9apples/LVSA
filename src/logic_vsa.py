import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class N3:
    def __init__(self, weight, margin_samples):
        self.weight = weight
        self.margin = torch.mean(torch.sum(torch.abs(margin_samples) ** 3, dim=-1)).item()

    def forward(self, factors, reduction=True):
        norm = 0
        for f in factors:
            f3 = torch.sum(torch.abs(f) ** 3, dim=-1)
            norm += self.weight * torch.mean(torch.relu(f3 - self.margin))

        if reduction:
            num = len(factors)
            norm = norm / num

        return norm


def aggregate_var_emb(sp_emb_list):

    var_emb = torch.mean(torch.stack([sp_emb for sp_emb in sp_emb_list], dim=1), dim=1)

    b_sp_list = [torch.mean(torch.abs(sp_emb), dim=-1, keepdim=True) for sp_emb in sp_emb_list]
    b_sp = torch.mean(torch.stack(b_sp_list, dim=1), dim=1)

    b_var = torch.mean(torch.abs(var_emb), dim=1, keepdim=True)
    b_norm = b_sp / b_var
    
    var_emb = var_emb * b_norm
    
    return var_emb

class LogicVSA(nn.Module):
    
    def __init__(self, 
                nentity,
                nrelation,
                rank,
                hidden_dim,
                test_batch_size,
                query_name_dict,
                w_r=5,
                emb_path=None,
                init_size=1e-3):
        super(LogicVSA, self).__init__()

        self.nentity = nentity
        self.nrelation = nrelation
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.test_batch_size = test_batch_size
        self.query_name_dict = query_name_dict

        if emb_path is None:
            print('Initialize embeddings...')
            self.ent_emb = nn.Parameter(torch.rand(nentity, rank) * init_size)
            self.rel_emb = nn.Parameter(torch.rand(nrelation, rank) * init_size)

            margin_samples = torch.zeros_like(self.ent_emb)
        else:
            print('Loading pretrained embeddings %s...' % emb_path)
            self.ent_emb, self.rel_emb = torch.load(emb_path)
            self.ent_emb = self.ent_emb.requires_grad_(False)
            self.rel_emb = self.rel_emb.requires_grad_(False)

            margin_samples = self.ent_emb[:self.rank]
            margin_samples = self.split_half(margin_samples)
            margin_samples = torch.sqrt(margin_samples[0] ** 2 + margin_samples[1] ** 2)

            self.rank = self.ent_emb.shape[1]

        self.mlp_i = nn.Sequential(
                nn.Linear(self.rank, self.rank*2),
                nn.LeakyReLU(),
                nn.Linear(self.rank*2, self.rank),
            )

        self.mlp_d = nn.Sequential(
                nn.Linear(self.rank*2, self.rank*2),
                nn.LeakyReLU(),
                nn.Linear(self.rank*2, self.rank*2),
                nn.LeakyReLU(),
                nn.Linear(self.rank*2, self.rank)
            )

        self.mlp_n = nn.Sequential(
                nn.Linear(self.rank*2, self.rank*2),
                nn.LeakyReLU(),
                nn.Linear(self.rank*2, self.rank),
            )

        
        self.reg = N3(1e-2, margin_samples)

        self.w_fr_n = 1
        self.w_n = 1 
        self.w_r = w_r 
        self.w_v = 0.25 
        self.w_a = 1. 
        self.w_rel = 0.5
        self.w_rel_1p = 0.5

        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda()

    def split_half(self, c_emb):
        half_rank = int(self.rank/2)
        return c_emb[..., :half_rank], c_emb[..., half_rank:]

    def split(self, lhs_emb, rel_emb, rhs_emb):
        lhs = self.split_half(lhs_emb)
        rel = self.split_half(rel_emb)
        rhs = self.split_half(rhs_emb)
        return lhs, rel, rhs

    def get_rev_relation(self, rel_emb):
        rev_rel_emb = self.split_half(rel_emb)
        rev_rel_emb = torch.hstack((rev_rel_emb[0], -rev_rel_emb[1]))
        return rev_rel_emb

    def get_factors(self, lhs, rel, rhs):
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))
        return factors

    def score_o(self, lhs_emb, rel_emb, rhs_emb, return_factors=False):
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(-1, -2)
        score_2 = (lhs[1] * rel[0] + lhs[0] * rel[1]) @ rhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def pred_tail(self, lhs_emb, rel_emb):
        lhs = self.split_half(lhs_emb)
        rel = self.split_half(rel_emb)

        tail_re = lhs[0] * rel[0] - lhs[1] * rel[1]
        tail_im = lhs[1] * rel[0] + lhs[0] * rel[1]
        return torch.cat([tail_re, tail_im], dim=-1)
    
    def pred_rel(self, lhs_emb, rhs_emb):
        lhs = self.split_half(lhs_emb)
        rhs = self.split_half(rhs_emb)

        rel_re = lhs[0] * rhs[0] + lhs[1] * rhs[1]
        rel_im = lhs[0] * rhs[1] - lhs[1] * rhs[0]
        return torch.cat([rel_re, rel_im], dim=-1)
    
    def score_tail(self, pred_emb, rhs_emb, all=True):
        if all:
            pred_scores1 = pred_emb[0] @ rhs_emb[0].transpose(-1, -2)
            pred_scores2 = pred_emb[1] @ rhs_emb[1].transpose(-1, -2)
        else:
            pred_scores1 = torch.sum(pred_emb[0] * rhs_emb[0], dim=-1)
            pred_scores2 = torch.sum(pred_emb[1] * rhs_emb[1], dim=-1)
        pred_scores = pred_scores1 + pred_scores2
        return pred_scores
    
    def pred_score(self, lhs_emb, rel_emb, rhs_emb):
        a_emb = self.pred_tail(lhs_emb, rel_emb)
        a_emb = self.split_half(a_emb)
        pred_scores = self.score_tail(a_emb, rhs_emb)
        return a_emb, pred_scores

    def encode(self, emb_list):
        vi_emb = self.mlp_d(torch.hstack(emb_list))
        return vi_emb
    
    def cal_reg_loss(self, factor_list, reduction=True):
        factors = []
        for factor in factor_list:
            factor_term = self.split_half(factor)
            factors += [torch.sqrt(factor_term[0] ** 2 + factor_term[1] ** 2)]
        l_reg = self.reg.forward(factors, reduction=reduction)
        return l_reg
    
    def cal_fit_rel_loss(self, h_emb, t_emb, r_embs, ans_rel):
        loss_fn = nn.CrossEntropyLoss(reduction='mean')

        a_emb = self.pred_rel(h_emb, t_emb)
        a_emb = self.split_half(a_emb)
        pred_scores = self.score_tail(a_emb, r_embs)
        l_fit_r = loss_fn(pred_scores, ans_rel)
        return l_fit_r
    
    def cal_logical_reg(self, pred_t_emb, v_emb):
        pred_scores = self.score_tail(pred_t_emb, v_emb, all=False)
        l = - torch.mean(torch.log(torch.sigmoid(pred_scores)))
        return l

    def mix_augmentation(self, u, v, ratio=0.5):
        w = torch.rand(u.shape[0], device=u.device).unsqueeze(-1) * ratio
        mix_emb = w * u + (1 - w) * v
        return mix_emb

    def forward_p(self, samples, rev=False):

        if rev: samples = torch.flip(samples, dims=[1])
        
        e1 = samples[:, 0]
        r1 = samples[:, 1]
        r2 = samples[:, 2]

        e1_emb = self.ent_emb[e1]
        r1_emb = self.rel_emb[r1]
        r2_emb = self.rel_emb[r2]

        ans = samples[:, -1]
        ae_emb = self.ent_emb[ans]

        if rev:
            r1_emb = self.get_rev_relation(r1_emb)
            r2_emb = self.get_rev_relation(r2_emb)

        pred_t1_emb = self.pred_tail(e1_emb, r1_emb) # t1 emb
        
        rev_r2_emb = self.get_rev_relation(r2_emb)
        r2_head = self.mlp_i(rev_r2_emb)
        pred_r2_emb = self.pred_tail(r2_head, rev_r2_emb)
        
        r1_head = self.mlp_i(r1_emb)
        pred_r1_emb = self.pred_tail(r1_head, r1_emb)
        pred_t1_emb_mix = self.mix_augmentation(pred_r1_emb.detach(), pred_t1_emb, ratio=0.3)

        emb_list = [pred_t1_emb_mix, pred_r2_emb]
        v1_emb_1 = self.encode(emb_list)
        v1_emb = v1_emb_1

        # N3 regularization
        l_reg = self.cal_reg_loss([v1_emb, r2_head])

        # main loss
        candidate_emb = self.ent_emb
        candidate_emb = self.split_half(candidate_emb)
        a_emb_1, pred_scores_1 = self.pred_score(v1_emb_1, r2_emb, candidate_emb)
        a_emb = torch.hstack(a_emb_1)
        
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        l_fit_1 = loss_fn(pred_scores_1, ans)
        l_fit = l_fit_1

        # rel prediction
        candidate_emb = self.rel_emb
        candidate_emb = self.split_half(candidate_emb)
        if rev: candidate_emb = (candidate_emb[0], -candidate_emb[1])

        l_fit_rel = []
        l_fit_rel.append(self.cal_fit_rel_loss(e1_emb, v1_emb_1, candidate_emb, r1))
        l_fit_rel.append(self.cal_fit_rel_loss(v1_emb_1, ae_emb, candidate_emb, r2))
        l_fit_rel = torch.mean(torch.hstack(l_fit_rel))

        # logical constraints
        pred_t1_emb = self.split_half(pred_t1_emb)
        pred_r2_emb = self.split_half(pred_r2_emb)
        v1_emb_1 = self.split_half(v1_emb_1)
        ae_emb = self.split_half(ae_emb)

        # vector regularization
        v1_emb = self.split_half(v1_emb)
        l_v = torch.mean(torch.relu(self.score_tail(v1_emb, v1_emb) + 1))
        
        a_emb = self.split_half(a_emb)
        pred_scores = self.score_tail(a_emb, ae_emb)
        target = torch.arange(pred_scores.shape[1], device=pred_scores.device)
        l_a = loss_fn(pred_scores, target)

        loss = (l_fit, l_reg, l_v, l_a, l_fit_rel)

        return loss
    
    def forward_3p(self, samples, rev=False):
        
        if rev: samples = torch.flip(samples, dims=[1])

        e1 = samples[:, 0]
        r1 = samples[:, 1]
        r2 = samples[:, 2]
        r3 = samples[:, 3]

        e1_emb = self.ent_emb[e1]
        r1_emb = self.rel_emb[r1]
        r2_emb = self.rel_emb[r2]
        r3_emb = self.rel_emb[r3]

        ans = samples[:, -1]
        ae_emb = self.ent_emb[ans]

        if rev:
            r1_emb = self.get_rev_relation(r1_emb)
            r2_emb = self.get_rev_relation(r2_emb)
            r3_emb = self.get_rev_relation(r3_emb)

        pred_t1_emb = self.pred_tail(e1_emb, r1_emb) # t1 emb

        rev_r2_emb = self.get_rev_relation(r2_emb)
        r2_head = self.mlp_i(rev_r2_emb)
        pred_r2_emb = self.pred_tail(r2_head, rev_r2_emb)

        r1_head = self.mlp_i(r1_emb)
        pred_r1_emb = self.pred_tail(r1_head, r1_emb)
        pred_t1_emb_mix = self.mix_augmentation(pred_r1_emb.detach(), pred_t1_emb, ratio=0.3)

        emb_list = [pred_t1_emb_mix, pred_r2_emb]
        v1_emb_1 = self.encode(emb_list)
        v1_emb = v1_emb_1

        factors = [v1_emb, r2_head]

        pred_t2_emb_1 = self.pred_tail(v1_emb_1, r2_emb)

        rev_r3_emb = self.get_rev_relation(r3_emb)
        r3_head = self.mlp_i(rev_r3_emb)
        pred_r3_emb = self.pred_tail(r3_head, rev_r3_emb)

        r2_tail = self.mlp_i(r2_emb)
        pred_r2t_emb = self.pred_tail(r2_tail, r2_emb)
        pred_t2_emb_mix = self.mix_augmentation(pred_r2t_emb.detach(), pred_t2_emb_1, ratio=0.3)
        
        emb_list = [pred_t2_emb_mix, pred_r3_emb]
        v2_emb_11 = self.encode(emb_list)
        v2_emb = v2_emb_11

        factors += [v2_emb, r3_head]
        # N3 regularization
        l_reg = self.cal_reg_loss(factors)

        # main loss
        candidate_emb = self.ent_emb
        candidate_emb = self.split_half(candidate_emb)
        a_emb_11, pred_scores_11 = self.pred_score(v2_emb_11, r3_emb, candidate_emb)

        a_emb = torch.hstack(a_emb_11)

        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        l_fit_11 = loss_fn(pred_scores_11, ans)
        l_fit = l_fit_11

        # rel prediction
        candidate_emb = self.rel_emb
        candidate_emb = self.split_half(candidate_emb)
        if rev: candidate_emb = (candidate_emb[0], -candidate_emb[1])

        l_fit_rel = []
        l_fit_rel.append(self.cal_fit_rel_loss(e1_emb, v1_emb_1, candidate_emb, r1))
        l_fit_rel.append(self.cal_fit_rel_loss(v1_emb_1, v2_emb_11, candidate_emb, r2))
        l_fit_rel.append(self.cal_fit_rel_loss(v2_emb_11, ae_emb, candidate_emb, r3))
        l_fit_rel = torch.mean(torch.hstack(l_fit_rel))
        
        # logical constraints
        pred_t1_emb = self.split_half(pred_t1_emb)
        pred_r2_emb = self.split_half(pred_r2_emb)
        v1_emb_1 = self.split_half(v1_emb_1)
        ae_emb = self.split_half(ae_emb)

        # vector regularization
        v1_emb = self.split_half(v1_emb)
        l_v = torch.mean(torch.relu(self.score_tail(v1_emb, v1_emb) + 1))

        v2_emb = self.split_half(v2_emb)
        l_v = l_v + torch.mean(torch.relu(self.score_tail(v2_emb, v2_emb) + 1))
        l_v = l_v / 2
        
        a_emb = self.split_half(a_emb)
        pred_scores = self.score_tail(a_emb, ae_emb)
        target = torch.arange(pred_scores.shape[1], device=pred_scores.device)
        l_a = loss_fn(pred_scores, target)

        loss = (l_fit, l_reg, l_v, l_a, l_fit_rel)

        return loss

    def forward(self, x, score_rhs=True, score_rel=False, score_lhs=False):
        lhs = self.ent_emb[x[:, 0]]
        rel = self.rel_emb[x[:, 1]]
        rhs = self.ent_emb[x[:, 2]]

        lhs = self.split_half(lhs)
        rel = self.split_half(rel)
        rhs = self.split_half(rhs)

        rhs_scores, rel_scores = None, None
        if score_rhs:
            to_score_entity = self.ent_emb
            to_score_entity = self.split_half(to_score_entity)
            rhs_scores = (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_entity[1].transpose(0, 1)
            )

            rhs_scores_batch = (
                (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ rhs[1].transpose(0, 1)
            )


        if score_rel:
            to_score_rel = self.rel_emb
            to_score_rel = self.split_half(to_score_rel)
            rel_scores = (
                (lhs[0] * rhs[0] + lhs[1] * rhs[1]) @ to_score_rel[0].transpose(0, 1) +
                (lhs[0] * rhs[1] - lhs[1] * rhs[0]) @ to_score_rel[1].transpose(0, 1)
            )
        if score_lhs:
            to_score_lhs = self.ent_emb
            to_score_lhs = self.split_half(to_score_lhs)
            lhs_scores = (
                (rel[0] * rhs[0] + rel[1] * rhs[1]) @ to_score_lhs[0].transpose(0, 1) + 
                (rel[0] * rhs[1] - rel[1] * rhs[0]) @ to_score_lhs[1].transpose(0, 1)
            )

            lhs_scores_batch = (
                (rel[0] * rhs[0] + rel[1] * rhs[1]) @ lhs[0].transpose(0, 1) + 
                (rel[0] * rhs[1] - rel[1] * rhs[0]) @ lhs[1].transpose(0, 1)
            )

        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        if score_rhs and score_rel and score_lhs:
            l_fit_rhs = loss_fn(rhs_scores, x[:, 2])
            l_fit_lhs = loss_fn(lhs_scores, x[:, 0])
            l_fit_rel = loss_fn(rel_scores, x[:, 1])

            l_fit = l_fit_rhs + l_fit_lhs

            target = torch.arange(rhs_scores_batch.shape[0], device=rhs_scores_batch.device)
            l_a = loss_fn(rhs_scores_batch, target) + loss_fn(lhs_scores_batch, target)

            factors = [torch.hstack(lhs), torch.hstack(rel), torch.hstack(rhs)]
            l_reg = self.cal_reg_loss(factors)

            return l_fit, l_fit_rel, l_a, l_reg
        
        elif score_rhs and score_rel:
            l_fit_rhs = loss_fn(rhs_scores, x[:, 2])
            l_fit_rel = loss_fn(rel_scores, x[:, 1])
            
            l_fit = l_fit_rhs + self.w_rel_1p * l_fit_rel

            target = torch.arange(rhs_scores_batch.shape[0], device=rhs_scores_batch.device)
            l_a = loss_fn(rhs_scores_batch, target)

            factors = [torch.hstack(lhs), torch.hstack(rel), torch.hstack(rhs)]
            l_reg = self.cal_reg_loss(factors)
            return l_fit, l_a, l_reg
        elif score_lhs and score_rel:
            pass
        elif score_rhs and score_lhs:
            pass
        elif score_rhs:
            return rhs_scores, factors
        elif score_rel:
            return rel_scores, factors
        elif score_lhs:
            return lhs_scores, factors
        else:
            return None
    

    def forward_n(self, samples):
        r1 = samples[:, 1]
        e1 = samples[:, 0]

        r2 = samples[:, 3]
        e2 = samples[:, 2]

        r1_emb = self.rel_emb[r1]
        e1_emb = self.ent_emb[e1]

        r2_emb = self.rel_emb[r2]
        e2_emb = self.ent_emb[e2]

        ans = samples[:, -1]
        ae_emb = self.ent_emb[ans]

        r1_head = self.mlp_i(r1_emb).detach()

        pred_t1_emb = self.pred_tail(e1_emb, r1_emb) # t1 emb
        pred_t2_emb = self.pred_tail(e2_emb, r2_emb) # t2 emb
        
        pred_r1_emb = self.pred_tail(r1_head, r1_emb)
        pred_t1_emb_mix = self.mix_augmentation(pred_r1_emb.detach(), pred_t1_emb, ratio=0.3)

        n_pred_t2_emb = self.mlp_n(torch.hstack([pred_t1_emb_mix, pred_t2_emb]))
        a_emb = aggregate_var_emb([pred_t1_emb, n_pred_t2_emb])

        # main loss
        candidate_emb = self.ent_emb
        candidate_emb = self.split_half(candidate_emb)
        a_emb = self.split_half(a_emb)
        pred_scores = self.score_tail(a_emb, candidate_emb)

        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        l_fit = loss_fn(pred_scores, ans)

        l_n = torch.mean(torch.sum((n_pred_t2_emb + pred_t2_emb)**2, dim=-1))
        
        nn_pred_t2_emb = self.mlp_n(torch.hstack([pred_t1_emb_mix, n_pred_t2_emb.detach()]))
        l_n = l_n + torch.mean(torch.sum((nn_pred_t2_emb - pred_t2_emb)**2, dim=-1))
        l_n = l_n / 2
    
        n_pred_t2_emb = self.split_half(n_pred_t2_emb)
        ae_emb = self.split_half(ae_emb)
        l_fit_reg = self.cal_logical_reg(n_pred_t2_emb, ae_emb)

        l_fit_reg = [l_fit_reg]

        loss = (l_fit, l_n, l_fit_reg)

        return loss

    def infer(self,
                positive_sample,
                negative_sample,
                subsampling_weight,
                batch_queries_dict,
                batch_idxs_dict):
        all_idxs = []
        scores = None

        for query_structure, queries in batch_queries_dict.items():
            all_idxs.extend(batch_idxs_dict[query_structure])

            with torch.no_grad():

                graph_type = self.query_name_dict[query_structure]

                if graph_type == "1p":
                    r1 = queries[:, 1]
                    r1_emb = self.rel_emb[r1]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    a_emb_1 = self.pred_tail(e1_emb, r1_emb)

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb_1 = self.split_half(a_emb_1)
                    scores_1 = self.score_tail(a_emb_1, candidate_emb)

                    scores = scores_1

                elif graph_type == "2p":
                    r1, r2 = queries[:, 1], queries[:, 2]
                    r1_emb, r2_emb = self.rel_emb[r1], self.rel_emb[r2]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v1_emb_1 = self.encode(emb_list)

                    a_emb_1 = self.pred_tail(v1_emb_1, r2_emb)

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb_1 = self.split_half(a_emb_1)
                    scores_1 = self.score_tail(a_emb_1, candidate_emb)

                    scores = scores_1


                elif graph_type == "3p":
                    r1, r2, r3 = queries[:, 1], queries[:, 2], queries[:, 3]
                    r1_emb, r2_emb, r3_emb = self.rel_emb[r1], self.rel_emb[r2], self.rel_emb[r3]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v1_emb_1 = self.encode(emb_list)

                    pred_t2_emb_1 = self.pred_tail(v1_emb_1, r2_emb)

                    flip_r3_emb = self.split_half(r3_emb)
                    flip_r3_emb = torch.hstack((flip_r3_emb[0], -flip_r3_emb[1]))
                    r3_head = self.mlp_i(flip_r3_emb)
                    pred_r3_emb = self.pred_tail(r3_head, flip_r3_emb)
                    emb_list = [pred_t2_emb_1, pred_r3_emb]

                    v2_emb_11 = self.encode(emb_list)

                    a_emb_11 = self.pred_tail(v2_emb_11, r3_emb)

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb_11 = self.split_half(a_emb_11)

                    scores_11 = self.score_tail(a_emb_11, candidate_emb)

                    scores = scores_11

                elif graph_type == "2i":
                    
                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    s1_emb = self.ent_emb[e1]
                    p1_emb = self.rel_emb[r1]

                    s2_emb = self.ent_emb[e2]
                    p2_emb = self.rel_emb[r2]

                    sp1_emb = self.pred_tail(s1_emb, p1_emb)
                    sp2_emb = self.pred_tail(s2_emb, p2_emb)

                    mean_sp_emb = (sp1_emb + sp2_emb)/2
                    sp_emb = mean_sp_emb

                    candidates_emb = self.ent_emb
                    candidates_emb = self.split_half(candidates_emb)
                    sp_emb = self.split_half(sp_emb)
                    scores = self.score_tail(sp_emb, candidates_emb)

                elif graph_type == "3i":
        
                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    r3 = queries[:, 5]
                    e3 = queries[:, 4]

                    s1_emb = self.ent_emb[e1]
                    p1_emb = self.rel_emb[r1]

                    s2_emb = self.ent_emb[e2]
                    p2_emb = self.rel_emb[r2]

                    s3_emb = self.ent_emb[e3]
                    p3_emb = self.rel_emb[r3]

                    sp1_emb = self.pred_tail(s1_emb, p1_emb)
                    sp2_emb = self.pred_tail(s2_emb, p2_emb)
                    sp3_emb = self.pred_tail(s3_emb, p3_emb)

                    mean_sp_emb = torch.mean(torch.stack([sp1_emb, sp2_emb, sp3_emb], dim=1), dim=1)
                    sp_emb = mean_sp_emb

                    candidates_emb = self.ent_emb
                    candidates_emb = self.split_half(candidates_emb)
                    sp_emb = self.split_half(sp_emb)
                    scores = self.score_tail(sp_emb, candidates_emb)

                elif graph_type == "ip":
                    
                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    r3 = queries[:, 4]

                    e1_emb = self.ent_emb[e1]
                    r1_emb = self.rel_emb[r1]

                    e2_emb = self.ent_emb[e2]
                    r2_emb = self.rel_emb[r2]

                    r3_emb = self.rel_emb[r3]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)
                    pred_t2_emb = self.pred_tail(e2_emb, r2_emb)

                    mean_sp_emb = (pred_t1_emb + pred_t2_emb)/2
                    pred_t_emb = mean_sp_emb

                    flip_r3_emb = self.split_half(r3_emb)
                    flip_r3_emb = torch.hstack((flip_r3_emb[0], -flip_r3_emb[1]))
                    r3_head = self.mlp_i(flip_r3_emb)
                    pred_r3_emb = self.pred_tail(r3_head, flip_r3_emb)
                    emb_list = [pred_t_emb, pred_r3_emb]

                    v_emb_1 = self.encode(emb_list)

                    a_emb_1 = self.pred_tail(v_emb_1, r3_emb)
                    a_emb_1 = self.split_half(a_emb_1)

                    candidates_emb = self.ent_emb
                    candidates_emb = self.split_half(candidates_emb)

                    score_1 = self.score_tail(a_emb_1, candidates_emb)

                    scores = score_1

                elif graph_type == "pi":
                    
                    r1, r2 = queries[:, 1], queries[:, 2]
                    r1_emb, r2_emb = self.rel_emb[r1], self.rel_emb[r2]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    r3 = queries[:, 4]
                    r3_emb = self.rel_emb[r3]

                    e2 = queries[:, 3]
                    e2_emb = self.ent_emb[e2]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v1_emb_1 = self.encode(emb_list)

                    sp1_emb_1 = self.pred_tail(v1_emb_1, r2_emb)
                    sp2_emb = self.pred_tail(e2_emb, r3_emb)

                    a_emb_1 = (sp1_emb_1 + sp2_emb)/2

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb_1 = self.split_half(a_emb_1)

                    scores_1 = self.score_tail(a_emb_1, candidate_emb)
                    scores = scores_1

                elif graph_type == "2u-DNF":

                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    s1_emb = self.ent_emb[e1]
                    p1_emb = self.rel_emb[r1]

                    s2_emb = self.ent_emb[e2]
                    p2_emb = self.rel_emb[r2]

                    candidates_emb = self.ent_emb

                    sp1_emb = self.pred_tail(s1_emb, p1_emb)
                    sp2_emb = self.pred_tail(s2_emb, p2_emb)

                    candidates_emb = self.split_half(candidates_emb)
                    sp1_emb = self.split_half(sp1_emb)
                    sp2_emb = self.split_half(sp2_emb)

                    scores_1 = self.score_tail(sp1_emb, candidates_emb)
                    scores_2 = self.score_tail(sp2_emb, candidates_emb)
                    
                    scores = torch.maximum(scores_1, scores_2)

                elif graph_type == "up-DNF":

                    r1, r2, r3 = queries[:, 1], queries[:, 3], queries[:, 5]
                    r1_emb, r2_emb, r3_emb = self.rel_emb[r1], self.rel_emb[r2], self.rel_emb[r3]

                    e1, e2 = queries[:, 0], queries[:, 2]
                    e1_emb, e2_emb = self.ent_emb[e1], self.ent_emb[e2]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)
                    pred_t2_emb = self.pred_tail(e2_emb, r2_emb)

                    flip_r3_emb = self.split_half(r3_emb)
                    flip_r3_emb = torch.hstack((flip_r3_emb[0], -flip_r3_emb[1]))
                    r3_head = self.mlp_i(flip_r3_emb)
                    pred_r3_emb = self.pred_tail(r3_head, flip_r3_emb)

                    emb_list = [pred_t1_emb, pred_r3_emb]
                    v1_emb_1 = self.encode(emb_list)

                    a1_emb_1 = self.pred_tail(v1_emb_1, r3_emb)

                    emb_list = [pred_t2_emb, pred_r3_emb]
                    v2_emb_1 = self.encode(emb_list)

                    a2_emb_1 = self.pred_tail(v2_emb_1, r3_emb)

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a1_emb_1 = self.split_half(a1_emb_1)
                    scores1_1 = self.score_tail(a1_emb_1, candidate_emb)

                    a2_emb_1 = self.split_half(a2_emb_1)
                    scores2_1 = self.score_tail(a2_emb_1, candidate_emb)

                    scores = torch.max(torch.stack([scores1_1, scores2_1], dim=1), dim=1).values

                elif graph_type == "2in":

                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    e1_emb = self.ent_emb[e1]
                    r1_emb = self.rel_emb[r1]
                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    e2_emb = self.ent_emb[e2]
                    r2_emb = self.rel_emb[r2]
                    pred_t2_emb = self.pred_tail(e2_emb, r2_emb)
                    n_pred_t2_emb = self.mlp_n(torch.hstack([pred_t1_emb, pred_t2_emb]))
                    a_emb = aggregate_var_emb([pred_t1_emb, n_pred_t2_emb])

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb = self.split_half(a_emb)
                    scores = self.score_tail(a_emb, candidate_emb)

                elif graph_type == "3in":

                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    r3 = queries[:, 5]
                    e3 = queries[:, 4]

                    e1_emb = self.ent_emb[e1]
                    r1_emb = self.rel_emb[r1]
                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    e2_emb = self.ent_emb[e2]
                    r2_emb = self.rel_emb[r2]
                    pred_t2_emb = self.pred_tail(e2_emb, r2_emb)

                    e3_emb = self.ent_emb[e3]
                    r3_emb = self.rel_emb[r3]
                    pred_t3_emb = self.pred_tail(e3_emb, r3_emb)


                    mean_pred_t_emb = (pred_t1_emb + pred_t2_emb)/2
                    a_emb = mean_pred_t_emb

                    n_pred_t3_emb = self.mlp_n(torch.hstack([a_emb, pred_t3_emb]))
                    mean_pred_t_emb = aggregate_var_emb([a_emb, n_pred_t3_emb])
                    a_emb = mean_pred_t_emb

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb = self.split_half(a_emb)
                    scores = self.score_tail(a_emb, candidate_emb)

                elif graph_type == "inp":
                    
                    # ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp'
                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    r3 = queries[:, 5]

                    e1_emb = self.ent_emb[e1]
                    r1_emb = self.rel_emb[r1]

                    e2_emb = self.ent_emb[e2]
                    r2_emb = self.rel_emb[r2]

                    r3_emb = self.rel_emb[r3]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)
                    pred_t2_emb = self.pred_tail(e2_emb, r2_emb)

                    n_pred_t2_emb = self.mlp_n(torch.hstack([pred_t1_emb, pred_t2_emb]))
                    mean_pred_t_emb = aggregate_var_emb([pred_t1_emb, n_pred_t2_emb])
                    pred_t_emb = mean_pred_t_emb

                    flip_r3_emb = self.split_half(r3_emb)
                    flip_r3_emb = torch.hstack((flip_r3_emb[0], -flip_r3_emb[1]))
                    r3_head = self.mlp_i(flip_r3_emb)
                    pred_r3_emb = self.pred_tail(r3_head, flip_r3_emb)
                    emb_list = [pred_t_emb, pred_r3_emb]

                    v_emb_1 = self.encode(emb_list)

                    a_emb_1 = self.pred_tail(v_emb_1, r3_emb)
                    a_emb_1 = self.split_half(a_emb_1)

                    candidates_emb = self.ent_emb
                    candidates_emb = self.split_half(candidates_emb)

                    score_1 = self.score_tail(a_emb_1, candidates_emb)

                    scores = score_1

                elif graph_type == "pin":

                    # (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin'
                    r1, r2 = queries[:, 1], queries[:, 2]
                    r1_emb, r2_emb = self.rel_emb[r1], self.rel_emb[r2]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    r3 = queries[:, 4]
                    r3_emb = self.rel_emb[r3]

                    e2 = queries[:, 3]
                    e2_emb = self.ent_emb[e2]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v1_emb_1 = self.encode(emb_list)

                    sp1_emb_1 = self.pred_tail(v1_emb_1, r2_emb)

                    sp2_emb = self.pred_tail(e2_emb, r3_emb)

                    n_pred_sp11_emb = self.mlp_n(torch.hstack([sp1_emb_1, sp2_emb]))
                    a_emb_1 = aggregate_var_emb([n_pred_sp11_emb, sp1_emb_1])

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb_1 = self.split_half(a_emb_1)

                    scores_1 = self.score_tail(a_emb_1, candidate_emb)
                    scores = scores_1

                elif graph_type == "pni":
                    # (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni'
                    r1, r2 = queries[:, 1], queries[:, 2]
                    r1_emb, r2_emb = self.rel_emb[r1], self.rel_emb[r2]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    r3 = queries[:, 5]
                    r3_emb = self.rel_emb[r3]

                    e2 = queries[:, 4]
                    e2_emb = self.ent_emb[e2]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v1_emb_1 = self.encode(emb_list)

                    sp1_emb_1 = self.pred_tail(v1_emb_1, r2_emb)

                    sp2_emb = self.pred_tail(e2_emb, r3_emb)

                    n_pred_sp11_emb = self.mlp_n(torch.hstack([sp2_emb, sp1_emb_1]))
                    a_emb_1 = aggregate_var_emb([n_pred_sp11_emb, sp2_emb])

                    candidate_emb = self.ent_emb
                    candidate_emb = self.split_half(candidate_emb)

                    a_emb_1 = self.split_half(a_emb_1)

                    scores_1 = self.score_tail(a_emb_1, candidate_emb)
                    scores = scores_1


                elif graph_type == "2pv" or graph_type == "2pf" or graph_type == "3pv1" or graph_type == "piv":
                    r1, r2 = queries[:, 1], queries[:, 2]
                    r1_emb, r2_emb = self.rel_emb[r1], self.rel_emb[r2]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v1_emb_1 = self.encode(emb_list)

                    candidate_emb = self.ent_emb

                    v1_emb_1 = self.split_half(v1_emb_1)
                    candidate_emb = self.split_half(candidate_emb)
                    scores_1 = self.score_tail(v1_emb_1, candidate_emb)

                    flip_v1_emb_1 = (v1_emb_1[0], -v1_emb_1[1])
                    candidate_emb = (candidate_emb[0], -candidate_emb[1])
                    scores_1 = scores_1 + self.score_tail(flip_v1_emb_1, candidate_emb)

                    scores = scores_1

                elif graph_type == "3pv2":
                    r1, r2, r3 = queries[:, 1], queries[:, 2], queries[:, 3]
                    r1_emb, r2_emb, r3_emb = self.rel_emb[r1], self.rel_emb[r2], self.rel_emb[r3]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v1_emb_1 = self.encode(emb_list)

                    pred_t2_emb_1 = self.pred_tail(v1_emb_1, r2_emb)

                    flip_r3_emb = self.split_half(r3_emb)
                    flip_r3_emb = torch.hstack((flip_r3_emb[0], -flip_r3_emb[1]))
                    r3_head = self.mlp_i(flip_r3_emb)
                    pred_r3_emb = self.pred_tail(r3_head, flip_r3_emb)
                    emb_list = [pred_t2_emb_1, pred_r3_emb]

                    v2_emb_11 = self.encode(emb_list)

                    candidate_emb = self.ent_emb

                    v2_emb_11 = self.split_half(v2_emb_11)
                    candidate_emb = self.split_half(candidate_emb)
                    scores_1 = self.score_tail(v2_emb_11, candidate_emb)

                    flip_v2_emb_11 = (v2_emb_11[0], -v2_emb_11[1])
                    candidate_emb = (candidate_emb[0], -candidate_emb[1])
                    scores_1 = scores_1 + self.score_tail(flip_v2_emb_11, candidate_emb)

                    scores = scores_1

                elif graph_type == "ipv":
                    r1 = queries[:, 1]
                    e1 = queries[:, 0]

                    r2 = queries[:, 3]
                    e2 = queries[:, 2]

                    r3 = queries[:, 4]

                    e1_emb = self.ent_emb[e1]
                    r1_emb = self.rel_emb[r1]

                    e2_emb = self.ent_emb[e2]
                    r2_emb = self.rel_emb[r2]

                    r3_emb = self.rel_emb[r3]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)
                    pred_t2_emb = self.pred_tail(e2_emb, r2_emb)

                    # # [[ORI]]
                    mean_sp_emb = (pred_t1_emb + pred_t2_emb)/2
                    pred_t_emb = mean_sp_emb

                    flip_r3_emb = self.split_half(r3_emb)
                    flip_r3_emb = torch.hstack((flip_r3_emb[0], -flip_r3_emb[1]))
                    r3_head = self.mlp_i(flip_r3_emb)
                    pred_r3_emb = self.pred_tail(r3_head, flip_r3_emb)
                    emb_list = [pred_t_emb, pred_r3_emb]

                    v_emb_1 = self.encode(emb_list)

                    candidate_emb = self.ent_emb

                    v_emb_1 = self.split_half(v_emb_1)
                    candidate_emb = self.split_half(candidate_emb)
                    scores_1 = self.score_tail(v_emb_1, candidate_emb)

                    flip_v_emb_1 = (v_emb_1[0], -v_emb_1[1])
                    candidate_emb = (candidate_emb[0], -candidate_emb[1])
                    scores_1 = scores_1 + self.score_tail(flip_v_emb_1, candidate_emb)

                    scores = scores_1

                elif graph_type == "upv1-DNF":
                    r1, r2 = queries[:, 1], queries[:, 5]
                    r1_emb, r2_emb = self.rel_emb[r1], self.rel_emb[r2]

                    e1 = queries[:, 0]
                    e1_emb = self.ent_emb[e1]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v_emb_1 = self.encode(emb_list)

                    candidate_emb = self.ent_emb

                    v1_emb_1 = self.split_half(v1_emb_1)
                    candidate_emb = self.split_half(candidate_emb)
                    scores_1 = self.score_tail(v1_emb_1, candidate_emb)

                    flip_v1_emb_1 = (v1_emb_1[0], -v1_emb_1[1])
                    candidate_emb = (candidate_emb[0], -candidate_emb[1])
                    scores_1 = scores_1 + self.score_tail(flip_v1_emb_1, candidate_emb)

                    scores = scores_1

                elif graph_type == "upv2-DNF":
                    r1, r2 = queries[:, 3], queries[:, 5]
                    r1_emb, r2_emb = self.rel_emb[r1], self.rel_emb[r2]

                    e1 = queries[:, 2]
                    e1_emb = self.ent_emb[e1]

                    pred_t1_emb = self.pred_tail(e1_emb, r1_emb)

                    flip_r2_emb = self.split_half(r2_emb)
                    flip_r2_emb = torch.hstack((flip_r2_emb[0], -flip_r2_emb[1]))
                    r2_head = self.mlp_i(flip_r2_emb)
                    pred_r2_emb = self.pred_tail(r2_head, flip_r2_emb)
                    emb_list = [pred_t1_emb, pred_r2_emb]

                    v_emb_1 = self.encode(emb_list)

                    candidate_emb = self.ent_emb

                    v1_emb_1 = self.split_half(v1_emb_1)
                    candidate_emb = self.split_half(candidate_emb)
                    scores_1 = self.score_tail(v1_emb_1, candidate_emb)

                    flip_v1_emb_1 = (v1_emb_1[0], -v1_emb_1[1])
                    candidate_emb = (candidate_emb[0], -candidate_emb[1])
                    scores_1 = scores_1 + self.score_tail(flip_v1_emb_1, candidate_emb)

                    scores = scores_1

                else:
                    raise ValueError(f'Unknown query type: {graph_type}')

        torch.cuda.empty_cache()
        return None, scores, None, all_idxs
    