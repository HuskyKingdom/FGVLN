import random
import optuna
import torch
import numpy as np
from utils.dataset.common import randomize_regions,randomize_tokens,pad_packed
import torch.nn.functional as F
import heapq

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
        positive_path_feature = None
        replace_feature = None
        i = 0
        
        for path in selected_paths:
            f, b, p, m = self.datasetIns._get_path_features(scan_id, path, heading)
            features.append(np.vstack(f))
            boxes.append(np.vstack(b))
            probs.append(np.vstack(p))
            masks.append(np.hstack(m))
            # store positive 
            positive_path_feature = (f,b,p,m) if i == 0 else positive_path_feature
            replace_feature = (f[-1],b[-1],p[-1],m[-1]) if i == 1 else replace_feature
            i += 1
        
        return features, boxes, probs, masks, path_id, instruction_index, positive_path_feature, replace_feature



    
    def wrap_features(self,features, boxes, probs, masks, path_id, instruction_index):

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



        # generate FGN_____________
        
        # get beam paths features
        beampaths = self.paths
        features, boxes, probs, masks, path_id, instruction_index,positive_path_feature,replace_feature  = self.get_selected_feature(beampaths)

        # sample M
        M = []

        for i in range(len(positive_path_feature[0])):
            m = trial.suggest_int(f"m_{i}",0,1)
            M.append(m)
        


        # unpack positive path features & replace to produce FGN
        for elem in range(len(positive_path_feature)):

            FGN = [None] * len(positive_path_feature[elem])

            for timestep in range(len(M)):
                FGN[timestep] = replace_feature[elem] if M[timestep] == 1 else positive_path_feature[elem][timestep]
            
            # append to positives
            if elem == 0:
                features.append(np.vstack(FGN))
            elif elem == 1:
                boxes.append(np.vstack(FGN))
            elif elem == 2:
                probs.append(np.vstack(FGN))
            elif elem == 3:
                masks.append(np.hstack(FGN))

    


        # compute objective_____________
        self.model.eval() # set to eval temporarly
        warped_features = self.wrap_features(features, boxes, probs, masks, path_id, instruction_index)
        outputs = self.model(*get_model_input(warped_features,self.device))

        target = warped_features[0]
        prediction = pad_packed(outputs["ranking"].squeeze(1), warped_features[13].cuda(device=self.device, non_blocking=True))
        prediction = prediction.unsqueeze(0)
        target = torch.tensor([target],device = self.device)

        loss = F.cross_entropy(prediction, target, ignore_index=-1)
        

        return loss


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

        self.max_trj_len = len(paths[0])
        self.iteration = iteration
        self.device = next(self.model.parameters()).device
        
    
    def find_n_best(self,all_,n): # return index of n best result

        values = []
        for trail in range(len(all_)):
            values.append(all_[trail].values[0])
        
        return heapq.nlargest(n, range(len(values)), key=values.__getitem__)


    def sample_fgn(self,num):

        M = []

        if self.type == 0:

            # random sample num times
            for i in range(num):
                temp_m = []
                for index in range(self.max_trj_len):
                    temp_m.append(random.randint(0,1))
                M.append(temp_m)

        else:

            # BO
            study = optuna.create_study(direction='maximize')
            study.optimize(Objective(self.paths,self.replace, self.model, self.datasetIns,self.beam_index,self.vln_index,self.target,self.device), n_trials=self.iteration)
            
            all_trials = study.trials
            best_idx = self.find_n_best(all_trials,num)

            # formating M and return
            for i in range(len(best_idx)):
                temp_m = []
                for index in range(self.max_trj_len):
                    temp_m.append(all_trials[i].params[f"m_{index}"])
                
                M.append(temp_m)

           
        return M