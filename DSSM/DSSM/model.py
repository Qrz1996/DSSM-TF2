import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid


class DSSM(Model):
    def __init__(self, feature_columns, item_features_list, embed_dim=16, l2_reg=1e-4):
        super(DSSM, self).__init__()

        self.feature_columns = feature_columns
        self.item_features_list = item_features_list
        self.embed_dim = embed_dim
        self.hidden_units = [self.embed_dim * 3, self.embed_dim * 2]
        self.l2_reg = l2_reg

        # embedding_layers
        self.user_embedding_layers = [Embedding(input_dim=feat['feat_num'],
                                                output_dim=self.embed_dim,
                                                input_length=1,
                                                embeddings_initializer='random_uniform',
                                                embeddings_regularizer=l2(self.l2_reg))
                                      for feat in self.feature_columns if feat['feat'] not in self.item_features_list]
        self.item_embedding_layers = [Embedding(input_dim=feat['feat_num'],
                                                output_dim=self.embed_dim,
                                                input_length=1,
                                                embeddings_initializer='random_uniform',
                                                embeddings_regularizer=l2(self.l2_reg))
                                      for feat in self.feature_columns if feat['feat'] in self.item_features_list]

        # user_tower
        self.user_tower = [Dense(units, activation='relu') for units in self.hidden_units]
        self.user_tower.append(Dense(self.embed_dim))

        # item_tower
        self.item_tower = [Dense(units, activation='relu') for units in self.hidden_units]
        self.item_tower.append(Dense(self.embed_dim))

        # output_layer
        # self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        user_inputs, item_inputs = inputs
        user_embedding = tf.concat([self.user_embedding_layers[i](user_inputs[:, i])
                                   for i in range(len(self.user_embedding_layers))], axis=-1)

        item_embedding = tf.concat([self.item_embedding_layers[i](item_inputs[:, i])
                                   for i in range(len(self.item_embedding_layers))], axis=-1)

        for user_tower_layer in self.user_tower:
            user_embedding = user_tower_layer(user_embedding)

        for item_tower_layer in self.item_tower:
            item_embedding = item_tower_layer(item_embedding)

        # calculate cousin similarity
        cos_similarity = self.cousin_similarity(user_embedding, item_embedding)

        # output = self.output_layer(cos_similarity)
        output = sigmoid(cos_similarity)

        return output, user_embedding, item_embedding
        # return output

    def cousin_similarity(self, user_embedding, item_embedding):
        ui = tf.reduce_sum(tf.multiply(user_embedding, item_embedding), axis=[1])
        user_normal = tf.sqrt(tf.reduce_sum(tf.square(user_embedding), axis=[1]))
        item_normal = tf.sqrt(tf.reduce_sum(tf.square(item_embedding), axis=[1]))

        return tf.expand_dims(tf.divide(ui, tf.multiply(user_normal, item_normal) + 1e-8), 1)


    def summary(self, line_length=None, positions=None, print_fn=None):
        user_inputs = Input(shape=(len(self.feature_columns) - len(self.item_features_list),))
        item_inputs = Input(shape=(len(self.item_features_list),))
        tf.keras.Model(inputs=[user_inputs, item_inputs],
                       outputs=self.call([user_inputs, item_inputs])).summary()

def test():
    spars_features = [{'feat': 'user', 'feat_num': 10},
                      {'feat': 'item', 'feat_num': 5},
                      {'feat': 'item_cate', 'feat_num': 3}]
    item_features_list = ['item', 'item_cate']
    model = DSSM(spars_features, item_features_list)
    model.summary()

# test()




