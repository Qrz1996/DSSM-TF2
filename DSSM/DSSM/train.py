import pickle
from model import DSSM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def train():
    path = r'/public/home/qrz/data/H&M/dssm_dataset.pkl'
    with open(path, 'rb') as f:
        train_x, train_y = pickle.load(f)
        feature_columns = pickle.load(f)

    embed_dim = 16
    learning_rate = 0.001
    batch_size = 1024
    epochs = 10

    item_features_list = ['article_id', 'product_type_no']

    model = DSSM(feature_columns, item_features_list, embed_dim=embed_dim)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy(), metrics=[AUC()])

    checkpoints_path = r'./checkpoints/DSSM_weights.epoch_{epoch:04d}.loss_{loss:.4f}.auc_{auc:.4f}.ckpt'
    checkpoint = ModelCheckpoint(filepath=checkpoints_path, save_freq='epoch', save_weights_only=True)
    history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint], shuffle=True)

    # plot train_loss
    plt.plot(len(history.history['loss']), history.history['loss'])
    plt.show()

def get_user_item_embeddings():
    learning_rate = 0.001
    path = r'/public/home/qrz/data/H&M/dssm_dataset.pkl'
    with open(path, 'rb') as f:
        train_x, train_y = pickle.load(f)
        feature_columns = pickle.load(f)
    # del train_x
    # del train_y
    item_features_list = ['article_id', 'product_type_no']
    embed_dim = 16

    with open(r'/public/home/qrz/data/H&M/get_embeddings_X.pkl', 'rb') as f:
        get_embeddings_X = pickle.load(f)

    model = DSSM(feature_columns, item_features_list, embed_dim=embed_dim)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy(), metrics=[AUC()])



    model.load_weights('./checkpoints/DSSM_weights.epoch_0007.loss_0.5865.auc_0.8296.ckpt')

    # model.fit(train_x, train_y)
    # _ = model.predict(get_embeddings_X)

    _, user_embeddings, item_embeddings = model.predict(get_embeddings_X)
    print(user_embeddings.shape, item_embeddings.shape)

    with open('/public/home/qrz/data/H&M/user_item_embeddings.pkl', 'wb') as f:
        pickle.dump((user_embeddings, item_embeddings), f, pickle.HIGHEST_PROTOCOL)
    print('嵌入向量已保存至/public/home/qrz/data/H&M/user_item_embeddings.pkl！')

if __name__ == '__main__':
    # train()
    get_user_item_embeddings()
