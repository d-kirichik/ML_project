import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer


class CustomModel(Model):

    def _build_layers_v2(self, parameters, outs, args):
        
        obs_real_obs = parameters["obs"]["real_obs"]
        fcnet_hiddens = args["fcnet_hiddens"]
        obs_action_mask = parameters["obs"]["action_mask"]
        
        for i, size in enumerate(fcnet_hiddens):
            label = "fc{}".format(i)
            obs_real_obs = slim.fully_connected(obs_real_obs, size, weights_initializer=normc_initializer(1.0), activation_fn=tf.nn.tanh, scope=label)
                
        action_logits = slim.fully_connected(obs_real_obs, outs, weights_initializer=normc_initializer(0.01), activation_fn=None, scope="fc_out")

        if outs == 1:
            return action_logits, obs_real_obs

        mask = tf.maximum(tf.log(obs_action_mask), tf.float32.min)
        logits = mask + action_logits

        return logits, obs_real_obs


def register_custom_model():
    ModelCatalog.register_custom_model("custom_model", CustomModel)