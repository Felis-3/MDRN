from models.base_model import Model
from common.utils import *
from common.GRU import *

class MDRN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, maxlen=30, maxlen_hard=20, use_negsampling=True):
        super(MDRN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                             ATTENTION_SIZE,
                                             use_negsampling, maxlen, maxlen_hard)

        is_training = True
        with tf.name_scope('UIM_layer'):
            self.f_eb = tf.einsum('bi,bj,bk->bijk', self.uid_batch_embedded, self.seq_hard_len_eb, self.trigger_eb)
            flat = tf.reshape(self.f_eb, [-1, 18 * 4 * 36])
            self.f_eb = tf.layers.dense(flat, 36, activation=tf.nn.relu)

            attention_output = din_attention_hian(self.f_eb, self.item_his_eb, self.time_stamp_his_batch_ph,
                                                  ATTENTION_SIZE, self.mask, stag='click_uim')
            att_fea = tf.reduce_sum(attention_output, 1)
            inp = tf.concat([self.uid_batch_embedded, self.trigger_eb, att_fea, self.item_his_eb_sum,
                             self.item_his_eb_sum * self.trigger_eb], 1)
            dnn1 = tf.layers.dense(inp, 200, activation=None, name='f1_uim')
            bn1 = tf.contrib.layers.batch_norm(dnn1, is_training=is_training, activation_fn=tf.nn.relu, scope='bn1_uim')

            dnn2 = tf.layers.dense(bn1, EMBEDDING_DIM * 2, activation=None, name='f2_uim')
            bn2 = tf.contrib.layers.batch_norm(dnn2, is_training=is_training, activation_fn=tf.nn.relu, scope='bn2_uim')
            dnn3 = tf.layers.dense(bn2, 2, activation=None, name='f3_uim')
            uim_logit = tf.nn.softmax(dnn3) + 0.00000001
            dnn3 = uim_logit[:,0:1]
            fusing_embedding = tf.multiply(dnn2, self.trigger_eb) + tf.multiply(1 - dnn2, self.item_eb)
            aux_loss_1 = - tf.reduce_mean(tf.log(uim_logit) * self.target_aux_ph)

        other_embedding_size = 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, other_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.item_his_eb)[0], 1])  # B*T,E
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.item_his_eb)[0], -1,
                                                                 self.position_his_eb.get_shape().as_list()[
                                                                     1]])  # B,T,E

        self.position_hard_his = tf.range(maxlen_hard)
        self.position_hard_embeddings_var = tf.get_variable("position_hard_embeddings_var",
                                                            [maxlen_hard, other_embedding_size])
        self.position_hard_his_eb = tf.nn.embedding_lookup(self.position_hard_embeddings_var,
                                                           self.position_hard_his)  # T,E
        self.position_hard_his_eb = tf.tile(self.position_hard_his_eb, [tf.shape(self.item_his_hard_eb)[0], 1])  # B*T,E
        self.position_hard_his_eb = tf.reshape(self.position_hard_his_eb, [tf.shape(self.item_his_hard_eb)[0], -1,
                                                                           self.position_hard_his_eb.get_shape().as_list()[
                                                                               1]])  # B,T,E

        time_stamp_expanded = tf.expand_dims(self.time_stamp_his_batch_ph, axis=-1)  # shape: [batch_size, seq_len, 1]
        time_stamp_expanded = tf.cast(time_stamp_expanded, dtype=tf.float32)
        inputs_with_time = tf.concat([self.item_his_eb, time_stamp_expanded],
                                     axis=-1)  # shape: [batch_size, seq_len, embedding_dim + 1]

        with tf.variable_scope("my_gru_scope"):
            cell = CustomGRUCell(num_units=36, alpha=1.0, beta=0.01, gamma=0.01, activation=tf.tanh)
            item_time_his_eb, final_state = tf.nn.dynamic_rnn(cell, inputs_with_time, dtype=tf.float32)

            multihead_attention_outputs12 = tf.compat.v1.layers.dense(item_time_his_eb, EMBEDDING_DIM * 4,
                                                                      activation=tf.nn.relu)
            multihead_attention_outputs12 = tf.compat.v1.layers.dense(multihead_attention_outputs12, EMBEDDING_DIM * 2)
            item_time_his_eb = multihead_attention_outputs12 + item_time_his_eb

        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputs = self_multi_head_attn(item_time_his_eb, num_units=EMBEDDING_DIM * 2,
                                                               num_heads=4, dropout_rate=0, is_training=True)
            print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
            multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4,
                                                                     activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.compat.v1.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        aux_loss_2 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], item_time_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1 + aux_loss_2

        with tf.name_scope("multi_hard_extractor_layer"):
            hard_outputs = tf.compat.v1.layers.dense(self.item_his_hard_eb, EMBEDDING_DIM * 4,
                                                     activation=tf.nn.relu)
            hard_outputs = tf.compat.v1.layers.dense(hard_outputs,
                                                     EMBEDDING_DIM * 2)
            hard_outputs_v2 = hard_outputs + self.item_his_hard_eb
            print('hard_outputs_v2.get_shape()', hard_outputs_v2.get_shape())
            with tf.name_scope('Attention_hard_layer'):
                attention_hard_output_tr, attention_hard_score_tr, attention_hard_scores_no_softmax_tr = din_attention_new(
                    fusing_embedding,
                    hard_outputs_v2,
                    self.position_hard_his_eb,
                    ATTENTION_SIZE,
                    self.mask_hard,
                    stag='tr' + '_v2')
                att_hard_fea_tr = tf.reduce_sum(attention_hard_output_tr, 1)

        with tf.name_scope("multi_extractor_layer"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36,
                                                                   num_heads=4, dropout_rate=0, is_training=True)
            multihead_attention_outputss = enhance_heads_with_cross_attention(multihead_attention_outputss)

            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs_v2,
                                                                         EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs3,
                                                                         EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
                with tf.name_scope('Attention_layer' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_new(self.item_eb,
                                                                                                       multihead_attention_outputs_v2,
                                                                                                       self.position_his_eb,
                                                                                                       ATTENTION_SIZE,
                                                                                                       self.mask,
                                                                                                       stag=str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)

                    att_fea_mix = tf.multiply(att_hard_fea_tr, dnn3) + tf.multiply(att_fea, 1 - dnn3)

                    if i == 0:
                        inp = att_fea_mix
                    else:
                        inp = tf.concat([inp, att_fea_mix], 1)

        with tf.name_scope("MLP"):
            Hadamard_fea = tf.multiply(self.trigger_eb, self.item_eb)
            cross_inp = tf.concat(
                [self.trigger_eb, self.item_eb, self.trigger_eb - self.item_eb, Hadamard_fea], 1)
            inp = tf.concat(
                [inp, self.uid_batch_embedded, self.item_his_eb_sum, cross_inp,
                 self.item_eb * self.item_his_eb_sum], -1)
            self.build_fcn_net(inp, use_dice=True)
