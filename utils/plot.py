import numpy as np
import matplotlib.pyplot as plt
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_scores_heatmap(model, epoch_num):
    # Extracting the scores from the model
    scores = []
    max_rows = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                scores.append(s1)
                max_rows = max(max_rows, s1.shape[0])

    # Check the shapes of the arrays in scores
    # shapes = [score.shape for score in scores]
    # if len(set(shapes)) > 1:
    #     print("Warning: Arrays in 'scores' have mismatched dimensions:", shapes)
    # Flatten and concatenate all scores tensors
    resized_scores = torch.cat([score.view(-1) for score in scores])

    # Reshape the concatenated scores tensor
    concatenated_scores = resized_scores.view(-1, len(resized_scores))
    print("the shape of concatenated_scores is: ", concatenated_scores.shape)
    scores_list = concatenated_scores[0].tolist()

    reshape_scores = np.array(scores_list).reshape(1, -1)
    print("the max score is: ", concatenated_scores.max())
    # print("the fist 10 scores are: ", scores_list[0:15])
    print("the min score is: ", concatenated_scores.min())
    print("total number of neurons: ", concatenated_scores.shape[1])

    # # Plotting the heatmap for scores
    plt.figure()
    plt.title('Score on neurons Heatmap on epoch {} attack cnn'.format(epoch_num))
    # plt.plot(concatenated_scores[:10].numpy())
    #
    plt.imshow(reshape_scores, cmap='viridis', aspect='auto', vmin=-0.07, vmax=0.07)
    plt.colorbar()
    plt.xlabel('Neuron')
    plt.ylabel('Importance score')
    plt.savefig('heatmap on epoch {} attack cnn.png'.format(epoch_num))
    # plt.savefig("line for score.png")



def plot_heatmap(score):
    plt.figure()
    plt.title('Score on neurons Heatmap attack cnn')
    plt.imshow(score, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Neuron')
    plt.ylabel('Importance score')
    plt.savefig('heatmap attack cnn.png')