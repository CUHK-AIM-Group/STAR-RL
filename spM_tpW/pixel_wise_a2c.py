import torch
from torch.autograd import Variable
from torch.distributions import Categorical

class PatchWiseAC_discrete_coordinate_classify:
    def __init__(self, config):
        self.gamma = config.gamma
        self.ac1_pi_loss_coeff = config.ac1_pi_loss_coeff
        self.ac1_v_loss_coeff = config.ac1_v_loss_coeff
        self.ac1_entropy_loss_coeff = config.ac1_entropy_loss_coeff

        self.use_discrete_action = config.use_discrete_action
        self.ac1_v_loss_use_l1 = config.ac1_v_loss_use_l1
        self.t = 0
        self.t_start = 0
        self.past_action_log_prob_row = {}
        self.past_action_entropy_row = {}
        self.past_action_log_prob_col = {}
        self.past_action_entropy_col = {}
        self.past_rewards = {}
        self.past_values = {}
        self.gamma_agent1 = config.gamma_agent1
        self.accumulate_reward_agent1 = config.accumulate_reward_agent1

    def reset(self):
        self.past_action_log_prob_row = {}
        self.past_action_entropy_row = {}
        self.past_action_log_prob_col = {}
        self.past_action_entropy_col = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = 0
        self.t = 0

    def compute_loss(self, reward=None):
        assert self.t_start < self.t
        self.past_rewards[self.t - 1] = reward
        R = 0
        v_loss = 0
        pi_loss_row = 0
        entropy_loss_row = 0
        pi_loss_col = 0
        entropy_loss_col = 0
        for i in reversed(range(self.t_start, self.t)):
            if self.accumulate_reward_agent1:
                R *= self.gamma_agent1
                R += self.past_rewards[i]
            else:
                R = self.past_rewards[i]
            # R *= self.gamma
            # R += self.past_rewards[i]
            # R = self.past_rewards[i]
            v = self.past_values[i]
            advantage = R - v.detach()
            # for row classification
            selected_log_prob_row = self.past_action_log_prob_row[i]
            entropy_row = self.past_action_entropy_row[i]
            # Log probability is increased proportionally to advantage
            pi_loss_row -= selected_log_prob_row * advantage
            # Entropy is maximized
            entropy_loss_row -= entropy_row

            # for col classification
            selected_log_prob_col = self.past_action_log_prob_col[i]
            entropy_col = self.past_action_entropy_col[i]
            # Log probability is increased proportionally to advantage
            pi_loss_col -= selected_log_prob_col * advantage
            # Entropy is maximized
            entropy_loss_col -= entropy_col

            # Accumulate gradients of value function
            if self.ac1_v_loss_use_l1: 
                v_loss += torch.abs(v - R)
            else:
                v_loss += (v - R) ** 2

        if self.ac1_v_loss_coeff != 1.0:
            v_loss *= self.ac1_v_loss_coeff
        
        if self.ac1_pi_loss_coeff != 1.0:
            pi_loss_row *= self.ac1_pi_loss_coeff
            pi_loss_col *= self.ac1_pi_loss_coeff

        if self.ac1_entropy_loss_coeff != 1.0:
            entropy_loss_row *= self.ac1_entropy_loss_coeff
            entropy_loss_col *= self.ac1_entropy_loss_coeff
        
        losses = dict()
        losses['v_loss'] = v_loss.view(pi_loss_row.shape).mean()
        # for row classification
        losses['pi_loss_row'] = pi_loss_row.mean()
        losses['entropy_loss_row'] = entropy_loss_row.mean()
        # for col classification
        losses['pi_loss_col'] = pi_loss_col.mean()
        losses['entropy_loss_col'] = entropy_loss_col.mean()

        self.reset()

        return losses 
        
    def act(self, value, reward, action=None, isTrain=False, deterministic=True):
        if isTrain:
            return self.act_and_train(action, value, reward)
        else:
            actions_row = self.determ_act(action[:,0,:], deterministic=deterministic)
            actions_col = self.determ_act(action[:,1,:], deterministic=deterministic)
            actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
            return actions#.cpu().numpy()

    def determ_act(self, pi, deterministic=True):
        if deterministic:
            _, actions = torch.max(pi.data, dim=1)
        else:
            pi = torch.clamp(pi.data, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi_reshape)
            actions = m.sample()
            actions = actions.view(n, h, w) 

        return actions#.cpu().numpy()

    def act_and_train(self, pi, value, reward):
        self.past_rewards[self.t - 1] = reward

        def randomly_choose_actions(pi):
            pi = torch.clamp(pi, min=0)
            n, num_actions = pi.shape
            # pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi.data)
            actions = m.sample()

            log_pi_reshape = torch.log(torch.clamp(pi, min=1e-9, max=1-1e-9))
            
            entropy = -torch.sum(pi * log_pi_reshape, dim=-1).view(n,1)#.view(n, 1, h, w)
            selected_log_prob = torch.gather(log_pi_reshape, 1, Variable(actions.unsqueeze(-1))).view(n, 1)#, h, w)
            actions = actions.view(n)#, 1)#h, w) 

            return actions, entropy, selected_log_prob
            
        # for the row classification
        pi_row = pi[:,0,:]
        actions_row, entropy, selected_log_prob = randomly_choose_actions(pi_row)
        self.past_action_log_prob_row[self.t] = selected_log_prob
        self.past_action_entropy_row[self.t] = entropy
        # for the col classification
        pi_col = pi[:,1,:]
        actions_col, entropy, selected_log_prob = randomly_choose_actions(pi_col)
        self.past_action_log_prob_col[self.t] = selected_log_prob
        self.past_action_entropy_col[self.t] = entropy

        # actions
        actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
        # set the values and time step
        self.past_values[self.t] = value
        self.t += 1
        return actions.cpu().numpy()


class PatchWiseAC_discrete:
    def __init__(self, config):
        self.gamma = config.gamma
        self.ac1_pi_loss_coeff = config.ac1_pi_loss_coeff
        self.ac1_v_loss_coeff = config.ac1_v_loss_coeff
        self.ac1_entropy_loss_coeff = config.ac1_entropy_loss_coeff

        self.use_discrete_action = config.use_discrete_action
        self.ac1_v_loss_use_l1 = config.ac1_v_loss_use_l1
        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_rewards = {}
        self.past_values = {}
        self.gamma_agent1 = config.gamma_agent1
        self.accumulate_reward_agent1 = config.accumulate_reward_agent1

    def reset(self):
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = 0
        self.t = 0

    def compute_loss(self, reward=None):
        assert self.t_start < self.t
        self.past_rewards[self.t - 1] = reward
        R = 0
        v_loss = 0
        pi_loss = 0
        entropy_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            if self.accumulate_reward_agent1:
                R *= self.gamma_agent1
                R += self.past_rewards[i]
            else:
                R = self.past_rewards[i]
            v = self.past_values[i]
            advantage = R - v.detach()
            selected_log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            # Log probability is increased proportionally to advantage
            pi_loss -= selected_log_prob * advantage
            # Entropy is maximized
            entropy_loss -= entropy
            # Accumulate gradients of value function
            if self.ac1_v_loss_use_l1: 
                v_loss += torch.abs(v - R)
            else:
                v_loss += (v - R) ** 2

        if self.ac1_v_loss_coeff != 1.0:
            v_loss *= self.ac1_v_loss_coeff
        
        if self.ac1_pi_loss_coeff != 1.0:
            pi_loss *= self.ac1_pi_loss_coeff

        if self.ac1_entropy_loss_coeff != 1.0:
            entropy_loss *= self.ac1_entropy_loss_coeff
        
        losses = dict()
        losses['pi_loss'] = pi_loss.mean()
        losses['v_loss'] = v_loss.view(pi_loss.shape).mean()
        losses['entropy_loss'] = entropy_loss.mean()

        self.reset()

        return losses 
        
    def act(self, value, reward, action=None, isTrain=False, deterministic=True):
        if isTrain:
            return self.act_and_train(action, value, reward)
        else:
            return self.determ_act(action, deterministic=deterministic)

    def determ_act(self, pi, deterministic=True):
        if deterministic:
            _, actions = torch.max(pi.data, dim=1)
        else:
            pi = torch.clamp(pi.data, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi_reshape)
            actions = m.sample()
            actions = actions.view(n, h, w) 

        return actions.cpu().numpy()

    def act_and_train(self, pi, value, reward):
        self.past_rewards[self.t - 1] = reward

        def randomly_choose_actions(pi):
            pi = torch.clamp(pi, min=0)
            n, num_actions = pi.shape
            # pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi.data)
            actions = m.sample()

            log_pi_reshape = torch.log(torch.clamp(pi, min=1e-9, max=1-1e-9))
            
            entropy = -torch.sum(pi * log_pi_reshape, dim=-1).view(n,1)#.view(n, 1, h, w)
            selected_log_prob = torch.gather(log_pi_reshape, 1, Variable(actions.unsqueeze(-1))).view(n, 1)#, h, w)
            actions = actions.view(n)#, 1)#h, w) 

            return actions, entropy, selected_log_prob

        actions, entropy, selected_log_prob = randomly_choose_actions(pi)
        
        self.past_action_log_prob[self.t] = selected_log_prob
        self.past_action_entropy[self.t] = entropy
        self.past_values[self.t] = value
        self.t += 1
        return actions.cpu().numpy()

class PatchWiseAC:
    def __init__(self, config):
        self.gamma = config.gamma
        self.ac1_v_loss_coeff = config.ac1_v_loss_coeff
        self.use_discrete_action = config.use_discrete_action
        self.t = 0
        self.t_start = 0
        self.past_rewards = {}
        self.past_values = {}

    def reset(self):
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = 0
        self.t = 0

    def compute_loss(self, reward=None):
        assert self.t_start < self.t
        self.past_rewards[self.t - 1] = reward
        R = 0
        v_loss = 0

        for i in reversed(range(self.t_start, self.t)):
            # R *= self.gamma
            # R += self.past_rewards[i]
            R = self.past_rewards[i]
            v = self.past_values[i]
            
            # Accumulate gradients of value function
            v_loss += (v - R) ** 2

        # if self.v_loss_coeff != 1.0:
            # v_loss *= self.v_loss_coeff

        losses = v_loss.mean() * self.ac1_v_loss_coeff

        self.reset()

        return losses 
        
    def act(self, value, reward, flag_updateCrt=False):
        self.past_rewards[self.t - 1] = reward
        self.past_values[self.t] = value
        self.t += 1
        return None


class PixelWiseA2C:
    """A2C: Advantage Actor-Critic.

    Args:
        model (A3CModel): Model to train
        gamma (float): Discount factor [0,1]
        beta (float): Weight coefficient for the entropy regularizaiton term.
        pi_loss_coeff(float): Weight coefficient for the loss of the policy
        v_loss_coeff(float): Weight coefficient for the loss of the value
            function
    """

    def __init__(self, config):

        self.gamma = config.gamma
        self.beta = config.beta
        self.pi_loss_coeff = config.pi_loss_coeff
        self.v_loss_coeff = config.v_loss_coeff

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_rewards = {}
        self.past_values = {}

    def reset(self):
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = 0
        self.t = 0

    def compute_loss(self):
        assert self.t_start < self.t
        R = 0

        pi_loss = 0
        v_loss = 0
        entropy_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            v = self.past_values[i]
            advantage = R - v.detach()
            selected_log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]
            # Log probability is increased proportionally to advantage
            pi_loss -= selected_log_prob * advantage
            # Entropy is maximized
            entropy_loss -= entropy
            # Accumulate gradients of value function
            v_loss += (v - R) ** 2

        if self.pi_loss_coeff != 1.0:
            pi_loss *= self.pi_loss_coeff

        if self.v_loss_coeff != 1.0:
            v_loss *= self.v_loss_coeff
	
        entropy_loss *= self.beta

        losses = dict()
        losses['pi_loss'] = pi_loss.mean()
        losses['v_loss'] = v_loss.view(pi_loss.shape).mean()
        losses['entropy_loss'] = entropy_loss.mean()
        return losses 

    def act_and_train(self, pi, value, reward):
        self.past_rewards[self.t - 1] = reward

        def randomly_choose_actions(pi):
            pi = torch.clamp(pi, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi_reshape.data)
            actions = m.sample()
        
            log_pi_reshape = torch.log(torch.clamp(pi_reshape, min=1e-9, max=1-1e-9))
            entropy = -torch.sum(pi_reshape * log_pi_reshape, dim=-1).view(n, 1, h, w)
        
            selected_log_prob = torch.gather(log_pi_reshape, 1, Variable(actions.unsqueeze(-1))).view(n, 1, h, w)
        
            actions = actions.view(n, h, w) 

            return actions, entropy, selected_log_prob

        actions, entropy, selected_log_prob = randomly_choose_actions(pi)
        
        self.past_action_log_prob[self.t] = selected_log_prob
        self.past_action_entropy[self.t] = entropy
        self.past_values[self.t] = value
        self.t += 1
        return actions.cpu().numpy()

    def act(self, pi, deterministic=True, store_reward=False, reward=None):
        if deterministic:
            _, actions = torch.max(pi.data, dim=1)
            if store_reward:
                self.past_rewards[self.t - 1] = reward
                self.t += 1
        else:
            pi = torch.clamp(pi.data, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi_reshape)
            actions = m.sample()
            actions = actions.view(n, h, w) 

        return actions.cpu().numpy()
    
    def get_reward(self, reward, done=False):
        self.past_rewards[self.t - 1] = reward
        if done:
            final_reward = self.compute_reward()
        else:
            raise Exception
        self.reset()
        return final_reward

    def compute_reward(self):
        assert self.t_start < self.t
        R = 0

        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
            
        return R 

    def stop_episode_and_compute_loss(self, reward, done=False):
        self.past_rewards[self.t - 1] = reward
        if done:
            losses = self.compute_loss()
        else:
            raise Exception
        self.reset()
        return losses
