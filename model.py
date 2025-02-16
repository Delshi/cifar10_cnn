import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard

from tensorboard import summary as summary_lib
from tensorboard.plugins import projector

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_curve

import io
import os
import itertools








TF_ENABLE_ONEDNN_OPTS = 0
BATCH_SIZE = 64








log_dir = 'logs' #* Если очень хочется поменять, придется редактировать .bat


model_summary_dir = log_dir + '\\' + 'model_summary'
model_summary_path = model_summary_dir + '\\' + 'model_summary.keras'

history_path = log_dir + '\\' + 'learning_history.npy'

graph_dir = log_dir + '\\graphs'

pr_curves_dir = log_dir + '\\pr_curves'

projector_dir = log_dir + '\\projector'
projector_config_path = projector_dir + '\\' + 'projector_config.pbtxt'
projector_embeddings_path = projector_dir + '\\' + 'embeddings.tsv'
projector_metadata_path = projector_dir + '\\' + 'metadata.tsv'








(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = y_train.reshape(-1, )

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']








def create_model():
    model = models.Sequential([
        layers.InputLayer((32, 32, 3), name = 'InputLayer'),

        layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.1),

        layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.15),

        layers.Conv2D(256, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3,3), padding = 'same', activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.GaussianDropout(0.01),
        layers.Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.15),

        layers.Flatten(),
        layers.Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),

        layers.GaussianNoise(0.04),
        layers.Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.05),

        layers.Dense(10, activation = 'softmax')
    ])

    model.compile(
        optimizer = optimizers.Adam(learning_rate = 0.001),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']        
    )

    model.summary()
    
    return model








datagen = ImageDataGenerator(
    rotation_range = 12,
    width_shift_range = 0.08,
    height_shift_range = 0.08,
    horizontal_flip = True,
    zoom_range = 0.08,
    fill_mode = 'nearest'
)








def log_images(epoch, images, labels, prefix, file_writer):
    num_samples = 9

    indices = np.random.choice(len(images), num_samples)

    selected_images = images[indices]
    selected_labels = labels[indices]

    fig = plt.figure(figsize = (8, 8))

    for i in range(num_samples):

        ax = fig.add_subplot(3, 3, i + 1)

        ax.imshow(selected_images[i])
        ax.axis('off')
        ax.set_title(f"Label: {classes[selected_labels[i].item()]}")

    buf = io.BytesIO()

    plt.savefig(buf, format = 'png')
    plt.close(fig)

    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels = 4)
    image = tf.expand_dims(image, 0)

    with file_writer.as_default():
        tf.summary.image(f"{prefix}_images", image, step = epoch)




def log_confusion_matrix(epoch, images, labels, model, file_writer):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis = 1)

    cm = confusion_matrix(labels, predicted_labels)

    fig = plt.figure(figsize = (8, 8))

    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment = "center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format = 'png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels = 4)
    image = tf.expand_dims(image, 0)

    with file_writer.as_default():
        tf.summary.image("confusion_matrix", image, step = epoch)




def log_pr_curves(epoch, images, labels, model, file_writer):
    predictions = model.predict(images)
    num_classes = len(classes)

    for class_id in range(num_classes):
        binary_labels = (labels == class_id).astype(int)
        class_predictions = predictions[:, class_id]

        precision, recall, _ = precision_recall_curve(binary_labels, class_predictions)

        fig = plt.figure(figsize = (8, 8))
        plt.plot(recall, precision, label = f'Class {classes[class_id]}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {classes[class_id]}')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format = 'png')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels = 4)
        image = tf.expand_dims(image, 0)

        with file_writer.as_default():
            tf.summary.image(f"pr_curve_class_{classes[class_id]}", image, step = epoch)




def smooth_curve(points, factor = 0.5):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points




def plot_training_history(history_data):
    plt.switch_backend('TkAgg')
    plt.figure(figsize = (12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(smooth_curve(history_data['loss']), label = 'Training Loss')
    plt.plot(smooth_curve(history_data['val_loss']), label = 'Validation Loss')
    plt.title('Loss Evolution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(smooth_curve(history_data['accuracy']), label = 'Training Accuracy')
    plt.plot(smooth_curve(history_data['val_accuracy']), label = 'Validation Accuracy')
    plt.title('Accuracy Evolution')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.switch_backend('Agg')




def plot_predictions(_):
    plt.switch_backend('TkAgg')

    plt.figure(figsize = (8, 8))
    indices = np.random.choice(len(X_test), 9)
    
    for i, idx in enumerate(indices):
        ax = plt.subplot(3, 3, i + 1)
        img = X_test[idx]
        true_label = y_test[idx][0]
        
        prediction = model.predict(img[np.newaxis, ...], verbose = 0)
        predicted_label = np.argmax(prediction)
        
        plt.imshow(img)
        plt.axis('off')
        
        color = 'green' if predicted_label == true_label else 'red'
        plt.text(5, 40, f"Expected: {classes[true_label]}", 
                color = 'black', fontsize = 12, backgroundcolor = 'white')
        plt.text(5, 44, f"Predicted: {classes[predicted_label]}", 
                color = color, fontsize = 12, backgroundcolor = 'white')
    
    plt.tight_layout()
    plt.show()

    plt.switch_backend('Agg')




def log_graph(graph_dir):
    graph_writer = tf.summary.create_file_writer(graph_dir)

    with graph_writer.as_default():
        tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = graph_dir)
    
    print(f"Graph logged.")




def log_projector(model, X_test, y_test, classes, projector_dir, projector_embeddings_path, projector_metadata_path):
    if not os.path.exists(projector_dir):
        os.mkdir(projector_dir)

    predictions = np.argmax(model.predict(X_test), axis=1)

    colors = {
        classes[0] : "#FF5733",
        classes[1] : "#33FF57",
        classes[2] : "#3357FF",
        classes[3] : "#F3FF33",
        classes[4] : "#FF33F6",
        classes[5] : "#33FFF3",
        classes[6] : "#A833FF",
        classes[7] : "#FF8F33",
        classes[8] : "#33FF8F",
        classes[9] : "#8F33FF" 
    }
    
    embeddings = model.predict(X_test)
    np.savetxt(projector_embeddings_path, embeddings, delimiter='\t')
    
    with open(projector_metadata_path, 'w') as f:
        f.write('Index\tLabel\tColor\n')
        for i, (pred, true) in enumerate(zip(predictions, y_test.flatten())):
            f.write(f"{i}\tPredicted: {classes[pred]} | True: {classes[true]}\t'{colors.get(classes[pred])}'\n")
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embeddings'
    embedding.metadata_path = "metadata.tsv"
    embedding.tensor_path = "embeddings.tsv"
    
    writer = tf.summary.create_file_writer(projector_dir)

    projector.visualize_embeddings(projector_dir, config)
    
    print(f"Projector saved.")




def log_gradients(epoch, model, X, y, writer):
    indices = np.random.choice(len(X), size = BATCH_SIZE, replace = False)
    sample_X = tf.convert_to_tensor(X[indices])
    sample_y = tf.convert_to_tensor(y[indices])
    
    with tf.GradientTape() as tape:
        predictions = model(sample_X, training = True)
        loss_fn = losses.get(model.loss)
        loss = loss_fn(sample_y, predictions)
    
    grads = tape.gradient(loss, model.trainable_weights)
    
    with writer.as_default():
        for grad, var in zip(grads, model.trainable_variables):
            if grad is not None:
                var_name = var.name.replace(':', '_')
                tf.summary.histogram(f'gradients/{var_name}', grad, step = epoch)




def on_epoch_end(epoch, logs):
    log_images(epoch, X_train, y_train, 'train', file_writer)
    log_images(epoch, X_test, y_test, 'validation', file_writer)
    log_confusion_matrix(epoch, X_test, y_test,  model, file_writer)
    log_pr_curves(epoch, X_test, y_test, model, file_writer)
    log_gradients(epoch, model, X_train, y_train, file_writer)

    print(f"Confusion Matrix, PR Curves, Gragients, Training and Validating images are logged.")



reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 6, verbose = 1)
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 15, restore_best_weights = True, verbose = 1)
tensorboard_cb = TensorBoard(log_dir = log_dir, histogram_freq = 1,  update_freq = 'epoch', write_graph = True, write_images = True)

callbacks = [reduce_lr, early_stop, tensorboard_cb, tf.keras.callbacks.LambdaCallback(on_epoch_end = on_epoch_end)]








#!=====================================MAIN LOOP=====================================
if (os.path.exists(log_dir)):
    print('Loading Model...')


    model = tf.keras.models.load_model(model_summary_path)

    model.summary()


    #*------------------------------------------------------------
    history_data = np.load(history_path, allow_pickle = True).item()
    #*------------------------------------------------------------


else:
    print('Start Learning...')


    model = create_model()


    #?============================================================
    file_writer = tf.summary.create_file_writer(log_dir)
    pr_curves_writer = tf.summary.create_file_writer(pr_curves_dir)
    graph_tracer = tf.summary.trace_on(True, profiler = False)
    #?============================================================


    #!=======================START================================
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size = BATCH_SIZE),
        epochs = 88,
        validation_data = (X_test, y_test),
        callbacks = callbacks,
    )
    #!========================STOP================================


    #?============================================================
    history_data = history.history

    log_graph( graph_dir)

    log_projector(model, 
                X_test, 
                y_test, 
                classes, 
                projector_dir, 
                projector_embeddings_path, 
                projector_metadata_path)
    #?============================================================


    #*------------------------------------------------------------
    os.mkdir(model_summary_dir)
    model.save(model_summary_path)

    np.save(history_path, history_data)
    #*------------------------------------------------------------
    

    print("Learning completed. Model saved.")
#!===================================END MAIN LOOP===================================








plot_training_history(history_data)
plot_predictions(None)