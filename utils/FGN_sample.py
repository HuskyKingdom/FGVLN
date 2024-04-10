import random
import optuna
import torch
import numpy as np
from utils.dataset.common import randomize_regions,randomize_tokens


def get_model_input(batch,device):
    (
        _,
        image_features,
        image_locations,
        image_mask,
        _,
        _,
        instr_tokens,
        instr_mask,
        _,
        _,
        segment_ids,
        co_attention_mask,
        _,
        opt_mask,
        _,
        attend_order_visual_feature,
    ) = batch

    # add bs dimension
    inputs = [
    image_features,
    image_locations,
    image_mask,
    instr_tokens,
    instr_mask,
    segment_ids,
    co_attention_mask,
    opt_mask,
    attend_order_visual_feature]

    opt_mask = opt_mask.unsqueeze(0)


    for i in range(len(inputs)):
        if isinstance(inputs[i], torch.Tensor):
            inputs[i] = inputs[i].unsqueeze(0)
            inputs[i] = inputs[i].cuda(device=device, non_blocking=True)
            


    # remove padding samples
    image_features = inputs[0][opt_mask] # 6 808 2048
    image_locations = inputs[1][opt_mask]
    image_mask = inputs[2][opt_mask]
    instr_tokens = inputs[3][opt_mask]
    instr_mask = inputs[4][opt_mask]
    segment_ids = inputs[5][opt_mask]
    
    # transform batch shape
    co_attention_mask = inputs[6].view(
        -1, inputs[6].size(2), inputs[6].size(3)
    )

    
    return (
        instr_tokens,
        image_features,
        image_locations,
        segment_ids,
        instr_mask,
        image_mask,
        co_attention_mask,
        None,
        attend_order_visual_feature,
    )



class Objective(object):
    def __init__(self, paths, replace, model,datasetIns,beam_index,vln_index,target,device):
        # Hold this implementation specific arguments as the fields of the class.
        self.positive_path = paths[0]
        self.paths = paths
        self.model = model
        self.replace = replace
        self.datasetIns = datasetIns
        self.beam_index = beam_index
        self.p_len = len(self.positive_path)
        self.vln_index = vln_index
        self.target = target
        self.device = device

    def get_selected_feature(self,selected_paths):

        vln_index = self.datasetIns._beam_to_vln[self.beam_index]
        vln_item = self.datasetIns._vln_data[vln_index]

        path_id, instruction_index = map(
            int, self.datasetIns._beam_data[self.beam_index]["instr_id"].split("_")
        )

        instr_tokens = torch.tensor(vln_item["instruction_tokens"][instruction_index])
        instr_mask = instr_tokens > 0
        instr_highlights = torch.tensor([])
        segment_ids = torch.zeros_like(instr_tokens)
        instr_highlights = torch.tensor([])

        scan_id = vln_item["scan"]
        heading = vln_item["heading"]

        # get path features
        features, boxes, probs, masks = [], [], [], []
        
        for path in selected_paths:
            f, b, p, m = self.datasetIns._get_path_features(scan_id, path, heading)
            features.append(np.vstack(f))
            boxes.append(np.vstack(b))
            probs.append(np.vstack(p))
            masks.append(np.hstack(m))
        
        return features, boxes, probs, masks, path_id, instruction_index



    
    def wrap_features(self,features, boxes, probs, masks, path_id, instruction_index):


        _ = None # ignored returns

        # get the order label of trajectory
        ordering_target = []
        order_atteneded_visual_feature = 1


        # convert data into tensors
        image_features = torch.from_numpy(np.array(features)).float()  # torch.tensor(features).float()
        image_boxes = torch.from_numpy(np.array(boxes)).float()
        image_probs = torch.from_numpy(np.array(probs)).float()
        image_masks = torch.from_numpy(np.array(masks)).long()
        instr_tokens = instr_tokens.repeat(len(features), 1).long()
        instr_mask = instr_mask.repeat(len(features), 1).long()
        segment_ids = segment_ids.repeat(len(features), 1).long()
        instr_highlights = instr_highlights.repeat(len(features), 1).long()

        # randomly mask image features
        if self.datasetIns._masked_vision:
            image_features, image_targets, image_targets_mask = randomize_regions(
                image_features, image_probs, image_masks
            )
        else:
            image_targets = torch.ones_like(image_probs) / image_probs.shape[-1]
            image_targets_mask = torch.zeros_like(image_masks)

        # randomly mask instruction tokens
        if self.datasetIns._masked_language:
            instr_tokens, instr_targets = randomize_tokens(
                instr_tokens, instr_mask, self.datasetIns._tokenizer, self.datasetIns.args
            )
        else:
            instr_targets = torch.ones_like(instr_tokens) * -1

        # construct null return items
        co_attention_mask = torch.zeros(
            2, self.datasetIns.args.max_path_length * self.datasetIns.args.max_num_boxes, self.datasetIns.args.max_instruction_length
        ).long()


        instr_id = torch.tensor([path_id, instruction_index]).long()

        target = torch.tensor(self.target).long()
        ordering_target = torch.tensor(ordering_target)
        instr_highlights = instr_highlights.repeat(len(features), 1).long()


        return (
            target, # ranking target
            image_features, # vit image features
            image_boxes, # vit image box features
            image_masks,
            image_targets,
            image_targets_mask,
            instr_tokens,
            instr_mask,
            instr_targets,
            instr_highlights,
            segment_ids,
            co_attention_mask,
            instr_id,
            torch.ones(image_features.shape[0]).bool(),
            _,
            order_atteneded_visual_feature,
        )
    


    def __call__(self, trial):

        # sample M
        M = []

        for i in range(self.p_len):
            m = trial.suggest_int(f"m_{i}",0,1)
            M.append(m)
        



        # generate FGN_____________
        
        # get beam paths features
        beampaths = self.paths
        features, boxes, probs, masks, path_id, instruction_index = self.get_selected_feature(beampaths)

        print(f" feature {torch.from_numpy(np.array(features)).float().shape} | box {torch.from_numpy(np.array(boxes)).float().shape} | prob {torch.from_numpy(np.array(probs)).float().shape} | mask {torch.from_numpy(np.array(masks)).float().shape}")
            
        
        FGN = [None] * len(self.positive_path)
        for i in range(len(M)):
            FGN[i] = self.replace if M[i] == 1 else self.positive_path[i]

        print(f"FGB {FGN}")

        # compute objective
        self.model.eval() # set to eval temporarly
        selected_paths = self.paths
        
        outputs = self.model(*get_model_input(self.get_selected_feature(selected_paths),self.device))

        print(f"out | {outputs}")
        

        return 0


class FGN_sampler:


    # sample type = 0 if random otherwise 1 for BO
    def __init__(self,paths,sample_type,replace,iteration,modle,datasetIns,beam_index,vln_index,target): 

        self.paths = paths
        self.positive = paths[0]
        self.type = sample_type
        self.replace = replace
        self.model = modle
        self.datasetIns = datasetIns
        self.beam_index = beam_index
        self.vln_index = vln_index
        self.target = target

        self.max_trj_len = 10
        self.iteration = iteration
        self.device = next(self.model.parameters()).device
        
    


    def sample_fgn(self,num):

        FGNs = []

        if self.type == 0:

            # random sample num times
            for i in range(num):
                current_FGN = self.positive[:]
                x_k = random.randint(0,self.max_trj_len)
                current_FGN[x_k] = self.replace
                FGNs.append(current_FGN)

        else:

            # BO
            study = optuna.create_study()
            study.optimize(Objective(self.paths,self.replace, self.model, self.datasetIns,self.beam_index,self.vln_index,self.target,self.device), n_trials=self.iteration)
            print(f"BEST PARAMETER ---------- {study.best_params}")
            pass

           

        return FGNs