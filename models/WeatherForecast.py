import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('du_lieu_thoi_tiet.csv')

# Tiền xử lý dữ liệu [-1, 1]
features = data[['Nhiệt độ thấp nhất', 'Nhiệt độ trung bình', 'Nhiệt độ cao nhất', 'Lượng mưa']].values
feature_mean = np.mean(features, axis=0)
feature_std = np.std(features, axis=0)
features = (features - feature_mean) / feature_std  # Chuẩn hóa dữ liệu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
num_train = 90
num_test = 8

train_data = features[:num_train]
test_data = features[num_train:num_train + num_test]


# Định nghĩa mô hình Generator
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=10),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(4, activation='relu')  # Sử dụng relu thay vì tanh
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return model


# Định nghĩa mô hình Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=4),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  metrics=['accuracy'])
    return model


# Định nghĩa mô hình GAN
def build_gan(generator, discriminator):
    model = tf.keras.Sequential([
        generator,
        discriminator
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    return model


# Khởi tạo mô hình
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)


# Huấn luyện mô hình GAN
def train_gan(epochs, batch_size=1):
    for epoch in range(epochs):
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_samples = train_data[idx]
        real_labels = np.ones((batch_size, 1))

        noise = np.random.normal(0, 1, (batch_size, 10))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

        noise = np.random.normal(0, 1, (batch_size, 10))
        g_loss = gan.train_on_batch(noise, real_labels)[0]

        print(
            f"{epoch}/{epochs} [D loss: {d_loss_real[0] + d_loss_fake[0]} | D accuracy: {0.5 * (d_loss_real[0] + d_loss_fake[0])}] [G loss: {g_loss}]")


# Thực hiện huấn luyện
train_gan(epochs=2000)


# Dự đoán dữ liệu mới
def generate_data(num_samples):
    noise = np.random.normal(0, 1, (num_samples, 10))
    generated_data = generator.predict(noise)
    # Khôi phục dữ liệu về phạm vi gốc
    generated_data = generated_data * feature_std + feature_mean
    return generated_data


# Dự đoán 8 tháng tiếp theo
predictions = generate_data(8)  # Dự đoán cho 8 tháng tiếp theo
predictions = pd.DataFrame(predictions,
                           columns=['Nhiệt độ thấp nhất', 'Nhiệt độ trung bình', 'Nhiệt độ cao nhất', 'Lượng mưa'])

# Làm tròn giá trị đến 2 chữ số thập phân
predictions = predictions.round(2)

# Khôi phục dữ liệu kiểm tra
test_data_original = test_data * feature_std + feature_mean
test_data_original = pd.DataFrame(test_data_original,
                                  columns=['Nhiệt độ thấp nhất', 'Nhiệt độ trung bình', 'Nhiệt độ cao nhất',
                                           'Lượng mưa'])

# Lưu dữ liệu dự đoán vào file CSV
predictions.to_csv('du_lieu_du_doan_8_thang.csv', index=False)


# So sánh dữ liệu dự đoán và dữ liệu kiểm tra
def plot_comparison(predictions, test_data_original):
    months = np.arange(1, 9)

    # Vẽ biểu đồ cho từng loại dữ liệu (nhiệt độ thấp nhất, trung bình, cao nhất, lượng mưa)
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    labels = ['Nhiệt độ thấp nhất', 'Nhiệt độ trung bình', 'Nhiệt độ cao nhất', 'Lượng mưa']

    for i, label in enumerate(labels):
        axs[i].plot(months, test_data_original[label], label='Test Data', marker='o', color='b')
        axs[i].plot(months, predictions[label], label='Generated Data', marker='x', color='r')
        axs[i].set_title(f'So sánh {label} (8 tháng)', fontsize=12)
        axs[i].set_xlabel('Tháng')
        axs[i].set_ylabel(label)
        axs[i].legend()

    plt.tight_layout()
    plt.show()


# Gọi hàm tạo biểu đồ
plot_comparison(predictions, test_data_original)
