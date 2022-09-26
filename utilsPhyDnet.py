import torch
import torch.nn as nn
import torch.nn.functional as F
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def fastFowardVideo(target, target_len):
    for index in range(target_len - 1):
        if index*2 < target_len:
            target[:, index, :, :, :] = target[:, index*2, :, :, :]
        else:
            target[:, index, :, :, :] = target[:, -1, :, :, :]
    return target

def differentLength(input, target, input_len, target_len):
    combined = torch.cat([input, target], dim=1)
    last_index = int((input_len + target_len) / 2)
    for index in range(input_len + target_len - 1):
        if index*2 < target_len + input_len:
            combined[:, index, :, :, :] = combined[:, index*2, :, :, :]
        else:
            combined[:, index, :, :, :] = combined[:, -1, :, :, :]

    return combined[:, :input_len, :, :, :], combined[:, input_len:last_index, :, :, :]


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}

    features_arr = []
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

def compute_style_loss(style_features, target_features):
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    style_grams = {layer: batch_gram_matrix(style_features[layer]) for layer in style_features}
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = batch_gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    return style_loss

def batch_gram_matrix(img):
    """
    Compute the gram matrix by converting to 2D tensor and doing dot product
    img: (batch, channel/depth, height, width)
    """
    b, d, h, w = img.size()
    img = img.view(b*d, h*w) # fix the dimension. It doesn't make sense to put b=1 when it's not always the case
    gram = torch.mm(img, img.t())
    return gram

def compute_content_loss(target_feature, content_feature):
    return torch.mean((target_feature - content_feature)**2)
