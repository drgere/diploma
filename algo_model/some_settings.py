import skopt
import losses
import similarity_accuracies


class settings():
    def __init__(self):
        self.los = losses.LOSSES
        self.acc = similarity_accuracies.ACCURACIES
        
        self.search_space = [
            skopt.space.Real(1e-6, 1e-1, "log-uniform", name="learning_rate"),
            skopt.space.Real(0.0, 1.0, "uniform", name="weight_decay"),
            skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_a"),
            skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_b"),
            skopt.space.Real(0.0, 0.9, "uniform", name="dropout"),
            skopt.space.Real(0.0, 1.0, "uniform", name="warmup_len"),
            skopt.space.Integer(1, 100, "log-uniform", name="batch_accum"),
        ]
        
        self.skopt_kwargs = {
            'n_calls': 5,
            'n_initial_points': 3
        }
        
    def crit(self):
        c1 = self.los['mse']
        c2 = self.los['cos']
        criterion = lambda pred, gt: c1(pred, gt) + 0.5*c2(pred, gt)
        return criterion
        
    def train_sum(self):
        train_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return train_summaries
        
    def dev_sum(self):
        dev_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return dev_summaries
        
    def score(self):
        return self.los['mse']
    
    def space(self):
        return self.search_space
    
    def sk_kwargs(self):
        return self.skopt_kwargs
    
    def train_input_handler(self):
        return lambda x,y: (x,)
    
    def train_output_handler(self):
        return lambda x,y: y
    
    def train_pred_handler(self):
        return lambda p: p
    
    def test_pred_handler(self):
        return lambda p, dataset: p

        
class embed_mse(settings):
    def __init__(self):
        super().__init__()
        
    def cr(self):
        return self.los['mse']
        
    def train_sum(self):
        train_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return train_summaries
        
    def dev_sum(self):
        dev_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return dev_summaries
        
    def score(self):
        return self.los['mse']
     
        
class embed_cos(embed_mse):
    def __init__(self):
        super().__init__()
        
    def cr(self):
        c1 = self.los['mse']
        c2 = self.los['cos']
        criterion = lambda pred, gt: c1(pred, gt) + 0.5*c2(pred, gt)
        return criterion


class def_base(settings):
    def __init__(self):
        super().__init__()
        
    def crit(self):
        return self.los['xen']
        
    def train_sum(self):
        train_summaries = {
            'xent': self.los['xen'],
            'xent_smooth': self.los['xens'],
            'acc': self.acc['accgls'],
        }
        return train_summaries
        
    def dev_sum(self):
        dev_summaries = {
            'xent': self.los['xen'],
            'xent_smooth': self.los['xens'],
            'acc': self.acc['accgls'],
        }
        return dev_summaries
        
    def score(self):
        return self.los['xen']
    
    def train_input_handler(self):
        return lambda x,y: (x, y[:-1])
    
    def train_output_handler(self):
        return lambda x, y: y.view(-1)
    
    def train_pred_handler(self):
        return lambda p: p.view(-1, p.size(-1))
    
    def test_pred_handler(self):
        return lambda p, dataset: dataset.decode(p)

    
class def_base(def_base):
    def __init__(self):
        super().__init__()
    
    def cr(self):
        return self.los['xens']
    
    def score(self):
        return self.los['xens']


    
SETTINGS = {
    'embed2embed-mlp-mse': embed_mse,
    'embed2embed-mlp-mse-cos5': embed_cos,
    'defmod-base-xen': def_base,
    'defmod-base-xens': def_base,
}
