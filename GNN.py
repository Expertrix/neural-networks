import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


dataset = Planetoid(root="data/Planetoid", name="Cora")
dataset = Planetoid(root="/tmp/Cora", name="Cora")

class CustomGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomGNN, self).__init__()
        self.layer1 = GCNConv(input_dim, hidden_dim)
        self.layer2 = GCNConv(hidden_dim, output_dim)

    def forward(self, feature_data, edge_info):
      
        x = self.layer1(feature_data, edge_info)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
      
        x = self.layer2(x, edge_info)
        return F.log_softmax(x, dim=1)



input_features = dataset.num_node_features
num_classes = dataset.num_classes
model = CustomGNN(input_features, 16, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

graph_data = dataset[0]  


def train_model():
    model.train()
    optimizer.zero_grad()
    output = model(graph_data.x, graph_data.edge_index)
    loss = F.nll_loss(output[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(200):
    loss_value = train_model()
    print(f"Epoch: {epoch+1:03d}, Loss: {loss_value:.4f}")


def evaluate_model():
    model.eval()
    with torch.no_grad():
        predictions = model(graph_data.x, graph_data.edge_index).argmax(dim=1)
        correct = (predictions[graph_data.test_mask] == graph_data.y[graph_data.test_mask]).sum()
        acc = int(correct) / int(graph_data.test_mask.sum())
    return acc


accuracy = evaluate_model()
print(f"Test Accuracy: {accuracy:.4f}")


"""
Epoch: 001, Loss: 1.9476
Epoch: 002, Loss: 1.8445
Epoch: 003, Loss: 1.7295
Epoch: 004, Loss: 1.5694
Epoch: 005, Loss: 1.4483
Epoch: 006, Loss: 1.2744
Epoch: 007, Loss: 1.1523
Epoch: 008, Loss: 1.0462
Epoch: 009, Loss: 0.9143
Epoch: 010, Loss: 0.8345
...
Epoch: 192, Loss: 0.0262
Epoch: 193, Loss: 0.0293
Epoch: 194, Loss: 0.0207
Epoch: 195, Loss: 0.0264
Epoch: 196, Loss: 0.0378
Epoch: 197, Loss: 0.0180
Epoch: 198, Loss: 0.0365
Epoch: 199, Loss: 0.0258
Epoch: 200, Loss: 0.0421
Test Accuracy: 0.8010
"""