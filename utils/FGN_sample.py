import random
import optuna




class Objective(object):
    def __init__(self, paths):
        # Hold this implementation specific arguments as the fields of the class.
        self.positive_path = paths[0]
        self.paths = paths
        self.p_len = len(self.positive_path)

    def __call__(self, trial):

        # sample M
        M = []

        for i in range(self.p_len):
            m = trial.suggest_int(0,1)
            M.append(m)
        
        # generate FGN
        return 0


class FGN_sampler:


    # sample type = 0 if random otherwise 1 for BO
    def __init__(self,paths,sample_type,replace,iteration): 

        self.paths = paths
        self.positive = paths[0]
        self.type = sample_type
        self.replace = replace


        self.max_trj_len = 10
        self.iteration = iteration

    


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
            study.optimize(Objective(self.paths), n_trials=self.iteration)
            print(study.best_params)
            pass

           

        return FGNs